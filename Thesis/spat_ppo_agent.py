# spat_ppo_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

# ---------------------------
# Neural Map Memory
# ---------------------------
class NeuralMapMemory(nn.Module):
    def __init__(self, env_size, map_channels=32, map_size=16, write_sigma=0.06):
        """
        write_sigma: in normalized map coordinates (0..1). 0.05 works well for map_size ~16-32.
        """
        super().__init__()
        self.map_channels = map_channels
        self.map_size = map_size
        self.env_size = env_size
        self.write_sigma = write_sigma 
        self.physical_sigma = 0.5 

        self.write_strength = nn.Parameter(torch.tensor(0.5))  

        gx = torch.linspace(0, 1, map_size)
        gy = torch.linspace(0, 1, map_size)
        grid_x, grid_y = torch.meshgrid(gx, gy, indexing='xy')
        self.register_buffer("grid_x", grid_x.clone())
        self.register_buffer("grid_y", grid_y.clone())
    
    def encode_positions(self, positions):
        """
        positions: (B,2) or (B,1,2) tensor expected to be normalized to [0,1].
        We clamp and ensure same device as buffers.
        """
        if positions is None:
            return None
        if positions.dim() == 3:
            positions = positions.squeeze(1)
        positions = positions.to(self.grid_x.device)
        return positions.clamp(0.0, 1.0)

    def gaussian_mask(self, positions):

        b = positions.size(0)
        gx = self.grid_x.unsqueeze(0).expand(b, -1, -1)
        gy = self.grid_y.unsqueeze(0).expand(b, -1, -1)

        px = positions[:, 0].view(b, 1, 1)
        py = positions[:, 1].view(b, 1, 1)

        sigma = float(self.write_sigma)
        sigma = max(1e-4, min(sigma, 1.0))
        dx = (gx - px ) ** 2
        dy = (gy - py ) ** 2

        mask = torch.exp(-(dx + dy) / (2 * sigma**2))
        return mask.unsqueeze(1)  # (B,1,H,W)

    def write(self, memory, write_vec, mask, positions, write_gate):

        if write_vec is None or positions is None:
            return memory

        B, C, H, W = memory.shape
        positions = self.encode_positions(positions)

        alpha = write_gate.view(B, 1, 1, 1)

        write_vec = write_vec.view(B, C, 1, 1)

        beta = 0.2
        memory = memory + beta * alpha * mask * (write_vec - memory)

        return memory

    def read(self, memory, positions, mask):

        if positions is None:
            pooled = F.adaptive_avg_pool2d(memory, 1).squeeze(-1).squeeze(-1)
            return pooled
        positions = self.encode_positions(positions)

        denom = mask.sum(dim=[2,3]) + 1e-6
        read_val = (memory * mask).sum(dim=[2,3]) / denom

        return read_val


# ---------------------------
# Agent Model
# ---------------------------
class AgentModel(nn.Module):
    def __init__(self, env_size = 10.0, obs_channels=6, map_channels=32, map_size=16, hidden_size=256, action_dim=None):
        super().__init__()

        self.map_size = map_size
        self.map_channels = map_channels
        self.debug_enabled = True
        self.env_size = env_size

        # obs encoder
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.obs_upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, map_channels, kernel_size=4, stride=2, padding=1),
        )   
        self.write_scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.write_fc = nn.Linear(self.map_channels, self.map_channels)
        self.write_gate_fc = nn.Linear(self.map_channels, 1)
        nn.init.constant_(self.write_gate_fc.bias, -2.0)

        self._prev_write_gate = None

        # memory
        self.memory_module = NeuralMapMemory(
            env_size=self.env_size,
            map_channels=map_channels,
            map_size=map_size,
            write_sigma=2.5 / map_size, #1.0/map_size
        )

        # projection + LSTM
        self.fc_proj_pi = nn.Linear(2 * map_channels, 256)
        self.fc_proj_v  = nn.Linear(2 * map_channels, 256)
        self.lstm_pi = nn.LSTM(256, hidden_size)
        self.lstm_v  = nn.LSTM(256, hidden_size)

        # actor/critic
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

        self.aux_value_head = nn.Linear(2 * map_channels, 1)

        self.apply(self._init_weights)

        self.total_env_steps = 0
        self.write_warmup_steps = 500

        self.just_triggered = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)  # Larger for value head
            nn.init.constant_(m.bias, 0)
    
    def init_states(self, batch_size, device):
        memory = torch.zeros(batch_size, self.map_channels, self.map_size, self.map_size, device=device)
        memory = memory + torch.randn_like(memory) * 0.01 
        h_pi = torch.zeros(1, batch_size, self.lstm_pi.hidden_size, device=device)
        c_pi = torch.zeros(1, batch_size, self.lstm_pi.hidden_size, device=device)
        h_v  = torch.zeros(1, batch_size, self.lstm_v.hidden_size,  device=device)
        c_v  = torch.zeros(1, batch_size, self.lstm_v.hidden_size,  device=device)
        self._prev_write_gate = None

        return {
            "memory": memory,
            "h_pi": h_pi,
            "c_pi": c_pi,
            "h_v":  h_v,
            "c_v":  c_v,
        }

    def get_max_write_gate(self):
        ep = self.total_env_steps //100

        if ep < 1000:
            return 1.0
        elif ep < 3000:
            return 1.0 - 0.7 * (ep - 1000) / 2000
        else:
            return 0.3

    def forward(self, obs, state, positions=None , disable_write=None):

        if positions is not None and positions.dim() == 3:
            positions = positions.squeeze(1)

        B = obs.size(0)

        # ---- encode observation ----
        conv = self.obs_encoder(obs)
        spatial_features = self.obs_upsample(conv)               
        spatial_features = F.adaptive_avg_pool2d(
                spatial_features, (self.map_size, self.map_size)
            )            
        
        # ---- global write vector ----
        mask = self.memory_module.gaussian_mask(positions)
        mask = mask / (mask.sum(dim=(2,3), keepdim=True) + 1e-6)

        local_feat = (spatial_features * mask).sum(dim=(2,3))
        write_vec = self.write_scale * torch.tanh(self.write_fc(local_feat))

        # ---- scalar write gate α ----
        alpha = torch.sigmoid(self.write_gate_fc(local_feat))
        alpha = torch.clamp(alpha, max=self.get_max_write_gate())
        self.last_alpha = alpha.mean()

        memory = state["memory"]

        if not disable_write and positions is not None:
            memory = self.memory_module.write(
                memory,
                write_vec,
                mask,
                positions,
                alpha
            )
        
        # ---- read from memory ----
        local_read = self.memory_module.read(memory, positions, mask)
        global_read = memory.mean(dim=(2,3))

        read_vec = torch.cat([local_read, global_read], dim=-1)

        policy_read = read_vec
        critic_read = read_vec

        proj_pi = torch.tanh(self.fc_proj_pi(policy_read))
        proj_v  = torch.tanh(self.fc_proj_v(critic_read))

        h_pi = state["h_pi"]
        c_pi = state["c_pi"]
        h_v  = state["h_v"]
        c_v  = state["c_v"]

        out_pi, (h_pi, c_pi) = self.lstm_pi(proj_pi.unsqueeze(0), (h_pi, c_pi))
        out_v,  (h_v,  c_v)  = self.lstm_v (proj_v.unsqueeze(0),  (h_v,  c_v))

        pi_logits = self.actor(out_pi.squeeze(0))
        value = self.critic(out_v.squeeze(0))

        state = {
            "memory": memory,
            "h_pi": h_pi,
            "c_pi": c_pi,
            "h_v":  h_v,
            "c_v":  c_v,
        }

        # Debug: Monitor memory statistics
        if self.debug_enabled and self.total_env_steps % 1000 == 0 and not self.just_triggered:
            with torch.no_grad():
                mem_std = memory.std().item()
                mem_abs = memory.abs().mean().item()
                
                # Check for saturation
                sat_frac = (memory.abs() > 4.5).float().mean().item()
                
                if sat_frac > 0.1:
                    print(f"⚠️ Memory saturation: {sat_frac:.2%} > 4.5")
                if mem_std < 0.005:
                    print(f"⚠️ Memory collapsing (std={mem_std:.5f})")
                if mem_abs > 3.0:
                    print(f"⚠️ Large memory magnitude: {mem_abs:.3f}")
        
            print(
                f"α={alpha.mean():.3f} | "
                f"mem_std={memory.std():.3f} | "
                f"local_norm={local_read.norm(dim=1).mean():.3f} | "
                f"global_norm={global_read.norm(dim=1).mean():.3f}"
                )

            self.just_triggered = True

        # Debug: Monitor write operations
        if not disable_write and write_vec is not None:
            with torch.no_grad():
                write_std = write_vec.std().item()
                if write_std > 2.0:
                    print(f"⚠️ Large write variance: {write_std:.3f}")

        return pi_logits, value, state, read_vec
    
    def freeze_policy_and_value_heads(self):
        for p in self.actor.parameters():
            p.requires_grad = False
        for p in self.critic.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


# ---------------------------
# PPO Agent
# ---------------------------
class PPOAgent:
    def __init__(self, obs_shape, action_dim, device, gamma=0.99, lam=0.95,
                 clip_ratio=0.08, lr=1e-4, max_grad_norm=0.3,
                 rollout_len=128, chunk_len=32, entropy_coef=0.01, value_coef=0.3, epochs=2,
                 write_warmup_steps=3000):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_len = rollout_len
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.epochs = epochs

        obs_channels = obs_shape[0]
        self.model = AgentModel(
            obs_channels=obs_channels,
            map_channels=32,
            map_size=16,
            hidden_size=256,
            action_dim=action_dim,
        ).to(device)

        # expose warmup config on model
        self.model.total_env_steps = 0
        self.model.write_warmup_steps = write_warmup_steps
        self.current_step = 0
        self.base_lr = lr

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-8)

        self.lr_decayed = False
        self.min_lr = 3e-5


        # Entropy scheduling
        self.entropy_init = 0.005     
        self.entropy_final = 0.003
        self.entropy_decay_updates = 600
        self.update_count = 0

        # buffers
        self.reset_buffers()

        # last logits stored (detached, on-device, shape (action_dim,))
        self._last_logits = None

    def reset_buffers(self):
        self.obs_buf = []
        self.pos_buf = []
        self.actions_buf = []
        self.logp_buf = []
        self.logits_buf = []
        self.rewards_buf = []
        self.done_buf = []
        self.values_buf = []
        self.hidden_buf = []
        self.mem_read_buf = []

    def act(self, obs, state, position=None, disable_write=False):
        # prepare obs
        if not isinstance(obs, torch.Tensor):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_t = obs.to(self.device).float()

        # prepare position
        if position is None:
            pos_in = None
        else:
            if not isinstance(position, torch.Tensor):
                pos_in = torch.tensor(position, dtype=torch.float32, device=self.device)
            else:
                pos_in = position.to(self.device).float()

            if pos_in.dim() == 1:
                pos_in = pos_in.unsqueeze(0)  # (1,2)
                
        with torch.no_grad():
            pi_logits, value, state, mem_read = self.model(
                obs_t, state, positions=pos_in, disable_write=disable_write
            )

        logits = pi_logits.squeeze(0)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        logp = dist.log_prob(action)
        logits_cpu = logits.detach()


        return int(action.item()), logp.detach(), float(value.item()), state, logits_cpu, mem_read

    def store(self, obs, position, action, logp, logits, reward, done, value, hidden, mem_read, disable_write=False):
        # obs
        if isinstance(obs, torch.Tensor):
            obs_t = obs.detach().cpu()
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32).cpu()
        self.obs_buf.append(obs_t)


        # ---- ALWAYS STORE POSITION ----
        if position is None:
            p = torch.zeros(2, dtype=torch.float32)
        else:
            p = position.detach().cpu().float().view(-1)
        self.pos_buf.append(p)

        # action
        self.actions_buf.append(int(action))

        # ---- STORE OLD LOGP DIRECTLY ----
        self.logp_buf.append(logp.detach().cpu())
        self.logits_buf.append(logits)

        # reward/value/done
        self.rewards_buf.append(float(reward))
        self.done_buf.append(bool(done))
        self.values_buf.append(float(value))
        self.mem_read_buf.append(mem_read.detach().cpu())


        # ---- STORE FULL RECURRENT STATE ----
        self.hidden_buf.append({
            "state": {k: v.detach().cpu() for k, v in hidden["state"].items()},
            "episode_start": hidden["episode_start"],
            "disable_write": disable_write,
        })

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

    def update(self):

        kl_sum = 0.0
        kl_count = 0
        self.model.just_triggered = False

        T = len(self.obs_buf)
        if T < self.rollout_len:
            return None
       
        # ----------- CONVERT BUFFERS TO TENSORS -----------
        obs = torch.stack(self.obs_buf).to(self.device)    # [T, C, H, W]
        if len(self.pos_buf) > 0 and self.pos_buf[0] is not None:
            pos = torch.stack(self.pos_buf).to(self.device)  # [T, 2]
        else:
            pos = None

        actions = torch.tensor(self.actions_buf, dtype=torch.long, device=self.device)
        rewards = torch.tensor(self.rewards_buf, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.done_buf, dtype=torch.float32, device=self.device)
        old_logp = torch.stack(self.logp_buf).to(self.device)  # [T,]
        old_logits = torch.stack(self.logits_buf).to(self.device)
        values = torch.tensor(self.values_buf, dtype=torch.float32, device=self.device)  # [T,]

       # quick check inside update() after stacking old_logits & old_logp:
        with torch.no_grad():
            recomputed_old_logp = torch.distributions.Categorical(logits=old_logits).log_prob(actions)
            max_diff = (recomputed_old_logp.to(self.device) - old_logp).abs().max().item()
            if max_diff > 1e-4:
                print(f"[WARN] old_logits vs stored old_logp mismatch max_diff={max_diff:.6f}")
        if self.done_buf[-1]:
            last_value = 0.0
        else:
            with torch.no_grad():
                last_obs = self.obs_buf[-1].unsqueeze(0).to(self.device)
                last_pos = self.pos_buf[-1].unsqueeze(0).to(self.device)

                last_hidden = self.hidden_buf[-1]
                state = {
                    k: last_hidden["state"][k].to(self.device)
                    for k in last_hidden["state"]
                }

                _, last_value, _, _ = self.model(
                    last_obs,
                    state,
                    positions=last_pos,
                    disable_write=last_hidden["disable_write"]
                )
                last_value = float(last_value.item()) 

        # ----------- COMPUTE GAE / RETURNS -----------
        adv_np, returns_np = self.compute_gae_from_lists(
            values=self.values_buf, rewards=self.rewards_buf, dones=self.done_buf, last_value=last_value)
        adv = torch.tensor(adv_np, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = adv.clamp(-4.0, 4.0)

        returns = torch.tensor(returns_np, dtype=torch.float32, device=self.device)

        # ------- Rebuild episode indices (still needed for chunking) -------
        episodes = []
        ep_start = 0
        for t in range(T):
            if self.done_buf[t]:
                episodes.append((ep_start, t + 1))
                ep_start = t + 1
        if ep_start < T:
            episodes.append((ep_start, T))

        # bookkeeping for losses / statistics
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_chunks = 0

        # ----------- PER-EPOCH, PER-EPISODE CHUNKED UPDATES -----------
        for epoch in range(self.epochs):
            early_stop = False

            # iterate episodes in order (optionally you can shuffle episode order here)
            for (ep_start, ep_end) in episodes:
                if early_stop:
                    break

                ep_len = ep_end - ep_start
                if ep_len <= 0:
                    continue

                chunk_starts = list(range(ep_start, ep_end, self.chunk_len))

                for start in chunk_starts:
                    end = min(start + self.chunk_len, ep_end)

                    h = self.hidden_buf[start]

                    state = {
                        k: v.to(self.device)
                        for k, v in h["state"].items()
                    }

                    if h["episode_start"]:
                        state = self.model.init_states(batch_size=1, device=self.device)

                    # autoregressively run the model for this chunk
                    pi_logits_pred = []
                    values_pred = []

                    state["memory"] = state["memory"].detach()
                    for t in range(start, end):
                        inp_obs = obs[t].unsqueeze(0)  # (1, C, H, W)
                        inp_pos = pos[t].unsqueeze(0) if pos is not None else None

                        disable_write_t = self.hidden_buf[t]["disable_write"]
                        logits_t, value_t, state, _ = self.model(
                            inp_obs, state, positions=inp_pos, disable_write=disable_write_t
                        )

                        pi_logits_pred.append(logits_t)
                        values_pred.append(value_t)

                    # stack predictions
                    pi_logits_pred = torch.cat(pi_logits_pred, dim=0)        # [chunk, action_dim]
                    values_pred = torch.cat(values_pred, dim=0).squeeze(-1) # [chunk]

                    if torch.isnan(pi_logits_pred).any():
                        print("NaN logits detected — skipping chunk")
                        continue

                    # slice relevant training targets
                    actions_chunk = actions[start:end]
                    old_logp_chunk = old_logp[start:end]
                    adv_chunk = adv[start:end]
                    ret_chunk = returns[start:end]
                    old_v = values[start:end].to(self.device)

                    # compute losses
                    dist = torch.distributions.Categorical(logits=pi_logits_pred)
                    logp = dist.log_prob(actions_chunk)
                    old_logits_chunk = old_logits[start:end]
                    old_dist = torch.distributions.Categorical(logits=old_logits_chunk.detach())
                    kl_div = torch.distributions.kl.kl_divergence(old_dist, dist).mean()

                    kl_sum += kl_div.item()
                    kl_count += 1

                    if kl_div.item() < 0.005 and self.update_count > 1500:
                        continue


                    # policy loss (PPO clipped objective)
                    ratio = torch.exp(logp - old_logp_chunk)
                    surr1 = ratio * adv_chunk
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_chunk
                    policy_loss = -torch.min(surr1, surr2).mean()

                    v_pred = values_pred
                    v_pred_clipped = old_v + (v_pred - old_v).clamp(-0.2, 0.2)
                    v_loss_unclipped = (v_pred - ret_chunk).pow(2)
                    v_loss_clipped = (v_pred_clipped - ret_chunk).pow(2)
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    #value_loss = torch.clamp(value_loss, max=5.0)

                    #value_loss = 0.5 * (v_pred - ret_chunk).pow(2).mean()

                    entropy_loss = dist.entropy().mean()

                    mem_reads = torch.stack(self.mem_read_buf).to(self.device)
                    pred_aux_value = self.model.aux_value_head(mem_reads).squeeze(-1)
                    aux_value_loss = F.mse_loss(pred_aux_value, returns.detach().unsqueeze(-1))

                    alpha_reg = alpha_reg = 0.0002
                    alpha_penalty = alpha_reg * self.model.last_alpha

                    aux_coef = 0.1

                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss + aux_coef * aux_value_loss #+ alpha_penalty

                    aux_coef = max(0.0, 0.1 * (1.0 - self.update_count / 5000))

                    # backward step
                    self.optimizer.zero_grad()
                    loss.backward()

                    clipped_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    #if clipped_norm > self.max_grad_norm*10.0:
                        #print(f"[Grad spike] norm={clipped_norm:.2f}, "
                        #    f"entropy={entropy_loss.item():.3f}, "
                        #    f"value_loss={value_loss.item():.3f}")

                    with torch.no_grad():
                        value_error = (values - returns).abs().mean().item()
                        value_std = values.std().item()
                        return_std = returns.std().item()
                        
                        if value_error > 2.0:
                            print(f"⚠️ Value function diverging: error={value_error:.3f}, "
                                f"value_std={value_std:.3f}, return_std={return_std:.3f}")
                        
                        # Check for value function collapse
                        #if value_std < 0.01:
                            #print(f"⚠️ Value function collapsed (std={value_std:.4f})")

                    self.optimizer.step()

                    # accumulate stats
                    total_loss += loss.item()
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy_loss.item()
                    total_chunks += 1

        # ---- compute averaged stats (avoid div by 0) ----
        if total_chunks == 0:
            stats = None
        else:
            stats = {
                'policy_loss': total_policy_loss / total_chunks,
                'value_loss': total_value_loss / total_chunks,
                'entropy': total_entropy / total_chunks,
                'total_loss': total_loss / total_chunks,
                'num_chunks': total_chunks
            }

        self.update_count += 1

        progress = min(
            1.0,
            self.update_count / self.entropy_decay_updates
        )

        self.entropy_coef = (
            self.entropy_init * (1.0 - progress)
            + self.entropy_final * progress
        )

        # ---- Clear buffers for next rollout ----
        self.reset_buffers()

        return stats, values

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])