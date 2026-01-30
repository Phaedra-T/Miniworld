
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class GlobalReadNet(nn.Module):
    def __init__(self, map_channels, embed_dim=128):
        super().__init__()
        self.pre_proj = nn.Conv2d(map_channels, embed_dim, kernel_size=1)
        self.attn = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, 1, kernel_size=1),
        )
        self.out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh(),
        )

    def forward(self, memory):
        B, C, H, W = memory.shape
        x = self.pre_proj(memory)
        attn_logits = self.attn(x)
        attn = torch.softmax(attn_logits.view(B, -1), dim=-1).view(B, 1, H, W)
        pooled = (x * attn).sum(dim=(2, 3))
        return self.out(pooled)


class NeuralMapMemory(nn.Module):
    def __init__(self, map_channels, map_size, env_size):
        super().__init__()
        self.map_channels = map_channels
        self.map_size = map_size
        self.last_alpha = 0.0
        self.env_size = env_size

        gx = torch.linspace(0, 1, map_size)
        gy = torch.linspace(0, 1, map_size)
        grid_x, grid_y = torch.meshgrid(gx, gy, indexing="xy")
        self.register_buffer("grid_x", grid_x.clone())
        self.register_buffer("grid_y", grid_y.clone())

        self.writer_input_dim = 128 + 128 + 16 + 32
        self.wh_conv = nn.Conv2d(self.writer_input_dim, map_channels, kernel_size=1)
        self.pos_proj = nn.Linear(2, 32)
        self.alpha_proj = nn.Linear(self.writer_input_dim, 1)
        nn.init.constant_(self.alpha_proj.bias, -3.0)

        self.local_value_head = nn.Sequential(
            nn.Linear(map_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def get_local_mask(self, pos, sigma):
        sigma = max(float(sigma), 1e-3)
        B = pos.size(0)
        pos = torch.clamp(pos, 0.0, 1.0)
        px = pos[:, 0] * (self.map_size - 1)
        py = pos[:, 1] * (self.map_size - 1)

        gx = self.grid_x * (self.map_size - 1)
        gy = self.grid_y * (self.map_size - 1)

        dist2 = (gx.unsqueeze(0) - px.view(B, 1, 1)) ** 2 + \
                (gy.unsqueeze(0) - py.view(B, 1, 1)) ** 2

        mask = torch.exp(-dist2 / (2 * sigma ** 2)).unsqueeze(1)
        mask = mask / (mask.sum(dim=(2, 3), keepdim=True) + 1e-6)
        return mask

    def write(self, memory, positions, context_vector, sigma=1.5):
        B, C, H, W = memory.shape

        positions = torch.clamp(positions, 0.0, 1.0)
        pos_emb = self.pos_proj(positions)
        writer_input = torch.cat([context_vector, pos_emb], dim=-1)
        h_hat = torch.tanh(self.wh_conv(writer_input.view(B, -1, 1, 1).expand(-1, -1, H, W)))

        mask = self.get_local_mask(positions, sigma)

        raw_alpha = torch.sigmoid(self.alpha_proj(writer_input)).view(B, 1, 1, 1)
        diff = (h_hat - memory) * mask
        novelty = diff.pow(2).mean(dim=1, keepdim=True)
        novelty_gate = torch.sigmoid(5 * (novelty - 0.15))

        alpha = torch.clamp(0.05 + (raw_alpha * novelty_gate), max=0.3)
        self.last_alpha = alpha

        memory = memory * (1 - 0.001)
        memory = (1 - alpha * mask) * memory + (alpha * mask) * h_hat
        return memory

    def read(self, memory, positions, sigma=1.5):
        mask = self.get_local_mask(positions, sigma)
        return (memory * mask).sum(dim=(2, 3))

    def predict_distance(self, memory, positions):
        local_feat = self.read(memory, positions)
        return self.local_value_head(local_feat)


class AgentModel(nn.Module):
    def __init__(self, env_size, map_channels=64, map_size=32, action_dim=None):
        super().__init__()

        self.env_size = env_size
        self.map_size = map_size
        self.map_channels = map_channels

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
        )
        
        self.memory_module = NeuralMapMemory(map_channels, map_size, env_size=env_size)
        self.global_read_net = GlobalReadNet(map_channels, embed_dim=128)
        self.heading_proj = nn.Linear(2, 16)

        self.ego_frac = 0.25
        self.ego_crop_size = max(5, int(round(self.ego_frac * self.map_size)))
        self.ego_cnn = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
        )

        policy_in = 128 + 128 + 128 + 16
        self.policy_fc = nn.Sequential(
            nn.Linear(policy_in, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.ego_predictor = nn.Sequential(
            nn.Linear(128 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

        self.apply(self._init_weights)

        self.total_env_steps = 0
        self.write_warmup_steps = 500

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0)

    def init_states(self, batch_size, device):
        return {
            "memory": torch.full((batch_size, self.map_channels, self.map_size, self.map_size), 1e-3, device=device)
        }

    def get_egocentric_crop(self, memory, pos, head):
        B = memory.size(0)
        device = memory.device
        crop_fraction = self.ego_crop_size / self.map_size
        look_ahead_dist = crop_fraction * 0.35

        cos_h = torch.cos(head)
        sin_h = torch.sin(head)

        shift_x = pos[:, 0] + cos_h * look_ahead_dist
        shift_y = pos[:, 1] + sin_h * look_ahead_dist

        t_x = shift_x * 2 - 1
        t_y = shift_y * 2 - 1

        s = crop_fraction
        theta = torch.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] = s * cos_h
        theta[:, 0, 1] = s * sin_h
        theta[:, 0, 2] = t_x
        theta[:, 1, 0] = -s * sin_h
        theta[:, 1, 1] = s * cos_h
        theta[:, 1, 2] = t_y

        grid = F.affine_grid(
            theta,
            torch.Size((B, memory.size(1), self.ego_crop_size, self.ego_crop_size)),
            align_corners=False,
        )

        return F.grid_sample(memory, grid, align_corners=False, padding_mode="zeros")

    def forward(self, obs, state, pos=None, head=None, disable_write=False):
        B = obs.size(0)
        T = obs.size(1) if obs.dim() == 5 else 1

        memory = state["memory"]
        obs_seq = obs if obs.dim() == 5 else obs.unsqueeze(1)
        obs_seq = obs_seq.float()
        obs_seq = torch.nan_to_num(obs_seq, nan=0.0, posinf=255.0, neginf=0.0)
        if obs_seq.max() > 1.0:
            obs_seq = obs_seq / 255.0
        obs_seq = torch.clamp(obs_seq, 0.0, 1.0)
        obs_seq = obs_seq.permute(0, 1, 4, 2, 3)  # B,T,C,H,W
        obs_flat = obs_seq.reshape(B * T, obs_seq.size(2), obs_seq.size(3), obs_seq.size(4))
        feat_seq = self.obs_encoder(obs_flat).view(B, T, -1)

        pos_seq = torch.clamp(pos.view(B, T, 2), 0.0, 1.0)
        head_seq = head.view(B, T)
        head_seq = torch.nan_to_num(head_seq, nan=0.0, posinf=0.0, neginf=0.0)

        pi_list, v_list, ego_list = [], [], []

        for t in range(T):
            curr_pos = pos_seq[:, t]
            curr_head = head_seq[:, t] if head_seq is not None else torch.zeros(B, device=obs.device)
            feature_vector = feat_seq[:, t]

            global_read = self.global_read_net(memory)
            heading_raw = torch.stack([torch.sin(curr_head), torch.cos(curr_head)], dim=-1)
            heading_emb = self.heading_proj(heading_raw)

            ego_crop = self.get_egocentric_crop(memory, curr_pos, curr_head)
            ego_features = self.ego_cnn(ego_crop)
            ego_list.append(ego_features)
            
            feature_vector_weak = feature_vector * 0.2  
            mem_combined = torch.cat([global_read, ego_features, feature_vector_weak], dim=-1)
            policy_input = torch.cat([mem_combined, heading_emb], dim=-1)

            p_vec = self.policy_fc(policy_input)
            pi_list.append(self.actor(p_vec))
            v_list.append(self.critic(p_vec))

            if not disable_write:

                writer_context = torch.cat([feature_vector_weak, ego_features, heading_emb], dim=-1)
                memory = self.memory_module.write(memory, curr_pos, writer_context)

        pi = torch.stack(pi_list, dim=1) if T > 1 else pi_list[0]
        vals = torch.stack(v_list, dim=1) if T > 1 else v_list[0]
        ego_feats = torch.stack(ego_list, dim=1) if T > 1 else ego_list[0]

        return pi, vals, ego_feats, {"memory": memory}


class NeuralMapAgent:
    def __init__(self, env_size, action_dim, device, gamma=0.99, lam=0.95, lr=1e-4, max_grad_norm=0.1,
                 n_steps=128, entropy_coef=0.01, value_coef=0.3, write_warmup_steps=3000):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.action_dim = action_dim

        self.model = AgentModel(
            env_size=env_size,
            map_channels=64,
            map_size=32,
            action_dim=action_dim,
        ).to(device)

        self.model.total_env_steps = 0
        self.model.write_warmup_steps = write_warmup_steps
        self.base_lr = lr

        self.params = list(self.model.parameters())
        self.optimizer = optim.Adam(self.params, lr=self.base_lr, weight_decay=1e-5, eps=1e-8)

        self.reset_buffers()

    def reset_buffers(self):
        self.obs_buf = []
        self.pos_buf = []
        self.heading_buf = []
        self.actions_buf = []
        self.mem_buf = []
        self.rewards_buf = []
        self.values_buf = []
        self.done_buf = []
        self.dist_buf = []

    def act(self, obs, state, position=None, heading=None, disable_write=False):
        if not isinstance(obs, torch.Tensor):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_t = obs.to(self.device).float()

        if position is not None:
            if not isinstance(position, torch.Tensor):
                pos_in = torch.tensor(position, dtype=torch.float32, device=self.device)
            else:
                pos_in = position.to(self.device).float()
            if pos_in.dim() == 1:
                pos_in = pos_in.unsqueeze(0)
        else:
            pos_in = None

        if heading is not None:
            if not isinstance(heading, torch.Tensor):
                h_in = torch.tensor(heading, dtype=torch.float32, device=self.device)
            else:
                h_in = heading.to(self.device).float()
            if h_in.dim() == 0:
                h_in = h_in.view(1)
        else:
            h_in = None

        obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=255.0, neginf=0.0)
        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)

        if pos_in is not None and pos_in.dim() == 1:
            pos_in = pos_in.unsqueeze(0)
        if pos_in is not None:
            pos_in = torch.clamp(pos_in, 0.0, 1.0)

        if h_in is not None:
            if h_in.dim() == 0:
                h_in = h_in.view(1)
            elif h_in.dim() == 2:
                h_in = h_in.view(-1)
            h_in = torch.nan_to_num(h_in, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            pi_logits, value, _, state = self.model(
                obs_t, state, pos=pos_in, head=h_in, disable_write=disable_write
            )

        pi_logits = pi_logits.squeeze(0)
        pi_logits = torch.nan_to_num(pi_logits, nan=0.0, posinf=0.0, neginf=0.0)
        dist = Categorical(logits=pi_logits)
        action = dist.sample()
        logp = dist.log_prob(action)

        return action.cpu().item(), logp.detach(), value.squeeze().cpu().item(), state, pi_logits.detach()

    def store(self, obs, position, heading, action, reward, done, value, state_before, dist):
        self.obs_buf.append(obs.detach().cpu().float() if isinstance(obs, torch.Tensor) else torch.tensor(obs))
        self.pos_buf.append(position.detach().cpu() if isinstance(position, torch.Tensor) else torch.tensor(position))
        self.heading_buf.append(heading.detach().cpu() if isinstance(heading, torch.Tensor) else torch.tensor(heading))
        self.actions_buf.append(int(action))
        self.rewards_buf.append(float(reward))
        self.mem_buf.append(state_before["memory"].detach().cpu())
        self.done_buf.append(bool(done))
        self.values_buf.append(float(value))
        self.dist_buf.append(dist)

    def compute_gae_from_lists(self, values, rewards, dones, last_value):
        if len(rewards) == 0:
            return np.array([]), np.array([])

        values_np = np.array(values + [last_value], dtype=np.float32)
        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.float32)

        T = len(rewards_np)
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones_np[t]
            delta = rewards_np[t] + self.gamma * values_np[t + 1] * nonterminal - values_np[t]
            lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
            adv[t] = lastgaelam

        returns = adv + values_np[:-1]
        return adv, returns

    def update(self, next_obs, next_state, next_pos, next_head, disable_write):
        T = len(self.obs_buf)
        if T < self.n_steps:
            self.reset_buffers()
            return None

        with torch.no_grad():
            _, values, _, _ = self.model(
                next_obs, next_state, pos=next_pos, head=next_head, disable_write=True
            )
            last_value = values.squeeze().item()

        adv, returns = self.compute_gae_from_lists(
            self.values_buf, self.rewards_buf, self.done_buf, last_value
        )
        adv = torch.tensor(adv, device=self.device).float()
        returns = torch.tensor(returns, device=self.device).float()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_seq = torch.stack(self.obs_buf).to(self.device).float().unsqueeze(0)
        pos_seq = torch.stack(self.pos_buf).to(self.device).float().view(1, T, 2)
        head_seq = torch.stack(self.heading_buf).to(self.device).float().view(1, T)
        actions = torch.tensor(self.actions_buf, dtype=torch.long, device=self.device)

        initial_state = {"memory": self.mem_buf[0].to(self.device)}

        pi_logits, values, ego_feats, state = self.model(
            obs_seq,
            initial_state,
            pos=pos_seq,
            head=head_seq,
            disable_write=disable_write,
        )

        pi_logits = pi_logits.squeeze(0)
        values = values.squeeze()
        ego_feats = ego_feats.squeeze(0)

        final_memory = state["memory"]
        mem_norm = final_memory.norm().item()

        dist = Categorical(logits=pi_logits)
        logp = dist.log_prob(actions)
        policy_entropy = dist.entropy().mean()

        action_oh = F.one_hot(actions, num_classes=self.action_dim).float()
        ego_pred = self.model.ego_predictor(torch.cat([ego_feats[:-1], action_oh[:-1]], dim=-1))
        ego_target = ego_feats[1:].detach()
        pred_coef = 0.25 #max(0.15, 0.35 - 0.2 * (self.model.total_env_steps / 4e5))
        nonterminal = 1.0 - torch.tensor(self.done_buf[:-1], device=self.device, dtype=torch.float32).unsqueeze(-1)
        predictive_loss = (((ego_pred - ego_target) ** 2) * nonterminal).mean()

        policy_loss = -(logp * adv.detach()).mean()
        value_loss = (values - returns).pow(2).mean()

        pos_t = pos_seq.squeeze(0)
        pred_dist = self.model.memory_module.predict_distance(final_memory, pos_t).squeeze(-1)
        dist_target = torch.tensor(self.dist_buf[-T:], device=self.device, dtype=torch.float32)
        nonterminal = 1.0 - torch.tensor(self.done_buf, device=self.device, dtype=torch.float32)
        dist_loss = (F.smooth_l1_loss(pred_dist, dist_target, reduction="none") * nonterminal).mean()
        
        memory_structure_loss = torch.tensor(0.0, device=self.device)
        if T >= 8:  
            sample_size = min(16, T)
            indices = torch.randperm(T, device=self.device)[:sample_size]
            pos_sample = pos_t[indices]  
            

            mem_feats = self.model.memory_module.read(final_memory, pos_sample)  

            pos_dist = torch.cdist(pos_sample, pos_sample)  
            feat_dist = torch.cdist(mem_feats, mem_feats)

            pos_dist_norm = pos_dist / (pos_dist.max() + 1e-6)
            feat_dist_norm = feat_dist / (feat_dist.max() + 1e-6)

            memory_structure_loss = F.mse_loss(feat_dist_norm, pos_dist_norm)


        loss = (policy_loss +
                self.value_coef * value_loss -
                self.entropy_coef * policy_entropy +
                pred_coef * predictive_loss +
                0.05 * dist_loss +
                0.02 * memory_structure_loss)  


        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()

        self.reset_buffers()

        return {
            "loss": loss.item(),
            "p_loss": policy_loss.item(),
            "v_loss": value_loss.item(),
            "pred_loss": predictive_loss.item(),
            "entropy": policy_entropy.item(),
            "dist_loss": dist_loss.item(),
            "mem_norm": mem_norm,
            "mem_structure_loss": memory_structure_loss.item(),
            "grad_norm": grad_norm.item(),
        }

    def save(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
