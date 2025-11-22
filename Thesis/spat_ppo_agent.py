# spat_ppo_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

# ---------------------------
# ConvLSTM cell
# ---------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c


# ---------------------------
# Neural Map Memory
# ---------------------------
class NeuralMapMemory(nn.Module):
    def __init__(self, map_channels=32, map_size=32, write_sigma=1.5):
        """
        write_sigma: in normalized map coordinates (0..1). 0.05 works well for map_size ~16-32.
        """
        super().__init__()
        self.map_channels = map_channels
        self.map_size = map_size
        self.write_sigma = write_sigma

        self.convlstm = ConvLSTMCell(
            in_channels=map_channels,
            hidden_channels=map_channels
        )

        self.post_conv = nn.Sequential(
            nn.Conv2d(map_channels, map_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, map_channels // 4), map_channels),
            nn.ReLU()
        )

        # Precompute coordinate grids (normalized 0..1)
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
        # clamp to [0,1] to prevent out-of-range writes
        return positions.clamp(0.0, 1.0)

    def gaussian_mask(self, positions):
        """
        Return mask shape (B,1,H,W) with gaussian centered at positions in normalized coords.
        Uses self.write_sigma directly (normalized units).
        """
        b = positions.size(0)
        gx = self.grid_x.unsqueeze(0).expand(b, -1, -1)
        gy = self.grid_y.unsqueeze(0).expand(b, -1, -1)

        px = positions[:, 0].view(b, 1, 1)
        py = positions[:, 1].view(b, 1, 1)

        sigma = float(self.write_sigma)
        sigma = max(1e-4, min(sigma, 1.0))
        dx = (gx - px) ** 2
        dy = (gy - py) ** 2

        mask = torch.exp(-(dx + dy) / (2 * sigma**2))
        return mask.unsqueeze(1)  # (B,1,H,W)

    def write(self, memory, values, positions):
        """
        Write values into memory using gaussian mask centered at positions.
        NOTE: we do NOT clamp memory here to avoid crushing convLSTM dynamics.
        """
        if positions is None or values is None:
            return memory

        B, C, H, W = memory.shape
        positions = self.encode_positions(positions)  # (B,2)
        mask = self.gaussian_mask(positions).expand(-1, C, -1, -1)
        # Blend
        memory = (1 - mask) * memory + mask * values
        return memory

    def forward(self, memory, values=None, positions=None, convlstm_state=None):
        h, c = convlstm_state

        # Optional write (values/positions may be None to disable write)
        if values is not None and positions is not None:
            memory = self.write(memory, values, positions)

        # LSTM update: ensure h/c on same device as memory
        if h.device != memory.device:
            h = h.to(memory.device)
            c = c.to(memory.device)

        h, c = self.convlstm(memory, h, c)
        memory = self.post_conv(h)

        memory = torch.tanh(memory)

        return memory, (h, c)


# ---------------------------
# Agent Model
# ---------------------------
class AgentModel(nn.Module):
    def __init__(self, obs_channels=3, map_channels=16, map_size=16, hidden_size=128, action_dim=None):
        super().__init__()

        self.map_size = map_size
        self.map_channels = map_channels

        # obs encoder
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(obs_channels, 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, map_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )

        # memory
        self.memory_module = NeuralMapMemory(
            map_channels=map_channels,
            map_size=map_size,
            write_sigma=0.05
        )

        # projection + LSTM
        self.fc_proj = nn.Linear(map_channels, 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=1, batch_first=False)

        # actor/critic
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

        nn.init.xavier_uniform_(self.critic.weight)
        nn.init.constant_(self.critic.bias, 0.0)

        self.total_env_steps = 0
        self.write_warmup_steps = 500

    def init_states(self, batch_size, device):
        memory = torch.zeros(batch_size, self.map_channels, self.map_size, self.map_size, device=device)
        h_conv = torch.zeros(batch_size, self.map_channels, self.map_size, self.map_size, device=device)
        c_conv = torch.zeros(batch_size, self.map_channels, self.map_size, self.map_size, device=device)
        h_lstm = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        c_lstm = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        return memory, (h_conv, c_conv), (h_lstm, c_lstm)

    def forward(self, obs, memory, convlstm_state, lstm_state, positions=None):
        """
        positions: expected to be normalized to [0,1] (B,2) or (B,1,2) or None.
        The forward will disable writes if we are still in warmup.
        """
        B = obs.size(0)

        # encode
        x = self.obs_encoder(obs)
        x = F.interpolate(x, size=(self.map_size, self.map_size), mode='bilinear', align_corners=False)


        # ------- WRITE WARMUP GATE -------
        disable_write = False
        # If attributes not set, default to writing enabled
        if hasattr(self, "write_warmup_steps") and hasattr(self, "total_env_steps"):
            disable_write = (self.total_env_steps < self.write_warmup_steps)

        write_values = None if disable_write else x
        write_positions = None if disable_write else positions

        # update memory (values/positions may be None to disable write)
        memory, convlstm_state = self.memory_module(
            memory,
            values=write_values,
            positions=write_positions,
            convlstm_state=convlstm_state
        )

        #lstm
        pooled = F.adaptive_avg_pool2d(memory, (1, 1)).view(B, -1)
        proj = self.fc_proj(pooled)
        proj = proj.unsqueeze(0)  # shape (1, B, feat)

        h_lstm, c_lstm = lstm_state

        if h_lstm.device != proj.device:
            h_lstm = h_lstm.to(proj.device)
            c_lstm = c_lstm.to(proj.device)

        out, (h_lstm, c_lstm) = self.lstm(proj, (h_lstm, c_lstm))

        pi_logits = self.actor(out.squeeze(0))
        value = self.critic(out.squeeze(0))

        return pi_logits, value, memory, convlstm_state, (h_lstm, c_lstm)


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
            map_channels=16,
            map_size=16,
            hidden_size=256,
            action_dim=action_dim,
        ).to(device)

        # expose warmup config on model
        self.model.total_env_steps = 0
        self.model.write_warmup_steps = write_warmup_steps

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-8)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=200, min_lr=1e-6
        )

        # buffers
        self.reset_buffers()

        # last logits stored (detached, on-device, shape (action_dim,))
        self._last_logits = None

    def reset_buffers(self):
        self.obs_buf = []
        self.pos_buf = []
        self.actions_buf = []
        self.logp_buf = []
        self.rewards_buf = []
        self.done_buf = []
        self.values_buf = []
        self.hidden_buf = []

    def act(self, obs, memory, conv_lstm, lstm_state, position=None):
        """
        obs: tensor (1, C, H, W) or numpy array
        memory, conv_lstm, lstm_state: current recurrent states (from init_states or previous step)
        position: (1,2) normalized in [0,1] OR None
        Returns: action (int), logp (tensor), value (float), memory, conv_lstm, lstm_state
        """
        # prepare obs
        if not isinstance(obs, torch.Tensor):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_t = obs.to(self.device).float()

        # prepare position
        pos_in = None
        if position is not None:
            if isinstance(position, torch.Tensor):
                p = position.to(self.device).float()
            else:
                p = torch.tensor(position, dtype=torch.float32, device=self.device)

            if p.dim() == 1:
                pos_in = p.unsqueeze(0)   # make (1,2)
            else:
                pos_in = p.view(1, -1)

        # forward (no grad)
        with torch.no_grad():
            pi_logits, value, memory, conv_lstm, lstm_state = self.model(
                obs_t, memory, conv_lstm, lstm_state, positions=pos_in
            )

        # normalize stored logits shape -> 1D vector per time-step
        # pi_logits likely shape (1, action_dim); convert to (action_dim,)
        self._last_logits = pi_logits.squeeze(0).detach().cpu()  # CPU, detached, 1D

        # sample action using logits on device
        # convert logits back to device for sampling to avoid device mismatch
        logits_device = self._last_logits.to(self.device)
        dist = torch.distributions.Categorical(logits=logits_device)
        action = dist.sample()
        logp = dist.log_prob(action)

        # return action as int, logp (detached tensor on CPU), value as float, and updated states
        return int(action.item()), logp.detach(), float(value.detach().item()), memory, conv_lstm, lstm_state

    def store(self, obs, position, action, logp, reward, done, value, hidden):

        # obs
        if isinstance(obs, torch.Tensor):
            obs_t = obs.detach().cpu()
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32).cpu()
        self.obs_buf.append(obs_t)

        idx = len(self.obs_buf) - 1

        # ---- ALWAYS STORE POSITION ----
        if position is None:
            p = torch.zeros(1,2)   # keep sequence alignment
        else:
            if isinstance(position, torch.Tensor):
                p = position.detach().cpu().float()
            else:
                p = torch.tensor(position, dtype=torch.float32)
            #if p.dim()==1:
            #    p = p.unsqueeze(0)
        self.pos_buf.append(p)

        # action
        a = int(action)
        self.actions_buf.append(a)

        # ---- STORE OLD LOGP DIRECTLY ----
        self.logp_buf.append(logp.detach().cpu())

        # reward/value/done
        self.rewards_buf.append(float(reward))
        self.done_buf.append(bool(done))
        self.values_buf.append(float(value))

        # ---- STORE FULL RECURRENT STATE ----
        self.hidden_buf.append({
            "memory": hidden["memory"].detach().cpu(),
            "conv_h": hidden["conv_h"].detach().cpu(),
            "conv_c": hidden["conv_c"].detach().cpu(),
            "lstm_h": hidden["lstm_h"].detach().cpu(),
            "lstm_c": hidden["lstm_c"].detach().cpu(),
        })


    def compute_gae_from_lists(self, values, rewards, dones):
        """
        Compute GAE with correct bootstrap:
        - If last step was terminal (done=True) -> bootstrap with 0
        - Else bootstrap with last value
        values: list of floats (len T)
        rewards: list of floats (len T)
        dones: list of bools (len T)
        Returns: adv (np array), returns (np array)
        """
        if len(rewards) == 0:
            return np.array([]), np.array([])

        # bootstrap value: 0 if last step terminal else last estimated V
        last_done = bool(dones[-1])
        last_v = 0.0 if last_done else float(values[-1])

        values_np = np.array(values + [last_v], dtype=np.float32)
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
         
        T = len(self.obs_buf)

        if T < self.rollout_len:
            return

        # ----------- CONVERT BUFFERS TO TENSORS -----------
        obs = torch.stack(self.obs_buf).to(self.device)    
        if len(self.pos_buf) > 0 and self.pos_buf[0] is not None:
            pos = torch.stack(self.pos_buf).to(self.device)  
        else:
            pos = None 

        actions = torch.tensor(self.actions_buf, dtype=torch.long, device=self.device)
        rewards = torch.tensor(self.rewards_buf, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.done_buf, dtype=torch.float32, device=self.device)
        old_logp = torch.stack(self.logp_buf).to(self.device)
        values = torch.tensor(self.values_buf, dtype=torch.float32, device=self.device)



        # ----------- COMPUTE ADV + RETURN (GAE) -----------
        adv_np, returns_np = self.compute_gae_from_lists(
            values=self.values_buf, rewards=self.rewards_buf, dones=self.done_buf
        )
        adv = torch.tensor(adv_np, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns_np, dtype=torch.float32, device=self.device)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        # ---- Sequence slicing ----
        T_full = (T // self.chunk_len) * self.chunk_len
        num_chunks = T_full // self.chunk_len

        idxs = list(range(num_chunks))
        random.shuffle(idxs)

        for ep in range(self.epochs):

            random.shuffle(idxs)

            for chunk_idx in idxs:

                start = chunk_idx * self.chunk_len
                end = start + self.chunk_len

                adv_chunk = adv[start:end]
                adv_chunk = (adv_chunk - adv_chunk.mean()) / (adv_chunk.std() + 1e-8)

                # ---- Restore hidden ----
                h = self.hidden_buf[start]

                mem = h["memory"].to(self.device)
                conv = (h["conv_h"].to(self.device), h["conv_c"].to(self.device))    
                lstm = (h["lstm_h"].to(self.device), h["lstm_c"].to(self.device))    

                pi_logits_pred = []
                values_pred = []

                for t in range(start, end):
                    inp_obs = obs[t].unsqueeze(0)  
                    inp_pos = pos[t].unsqueeze(0) if pos is not None else None

                    logits_t, value_t, mem, conv, lstm = self.model(
                        inp_obs, mem, conv, lstm, positions=inp_pos
                    )
                    pi_logits_pred.append(logits_t)
                    values_pred.append(value_t)

                # stack predictions
                pi_logits_pred = torch.cat(pi_logits_pred, dim=0)   
                values_pred = torch.cat(values_pred, dim=0).squeeze(-1)  

                ret_chunk = returns[start:end]
                adv_chunk = adv[start:end]
                actions_chunk = actions[start:end]
                old_logp_chunk = old_logp[start:end]

                # new logprob
                dist = torch.distributions.Categorical(logits=pi_logits_pred)
                logp = dist.log_prob(actions_chunk)

                # ratio and policy loss
                ratio = torch.exp(logp - old_logp_chunk)
                surr1 = ratio * adv_chunk
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_chunk
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss (clipped)
                old_v = values[start:end].to(self.device)
                v_pred = values_pred
                v_pred_clipped = old_v + torch.clamp(v_pred - old_v, -self.clip_ratio, self.clip_ratio)
                v_loss_unclipped = (v_pred - ret_chunk).pow(2)
                v_loss_clipped = (v_pred_clipped - ret_chunk).pow(2)
                v_loss = torch.max(v_loss_unclipped, v_loss_clipped)  
                value_loss = 0.5 * v_loss.mean()

                entropy_loss = dist.entropy().mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss 

                self.optimizer.zero_grad()
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                # Log if gradients are exploding/vanishing
                if total_norm > 2.0:
                    print(f"Warning: High gradient norm: {total_norm:.4f}")
                if total_norm < 1e-4:
                    print(f"Warning: Vanishing gradients: {total_norm:.4f}")
                self.optimizer.step()


                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()

        # clear buffers
        self.reset_buffers()

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def reset_training_state(self, lr = 1e-4):
        # Reset learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Reset scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=200, min_lr=1e-6
        )
        
        # Reset any other training state
        self.reset_buffers()
        self.model.total_env_steps = 0  # Reset warmup counter if needed