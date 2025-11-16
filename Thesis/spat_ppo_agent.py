import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvLSTMCell(nn.Module):
    """Basic ConvLSTM cell. Input and hidden states are (B, C, H, W)."""
    def __init__(self, in_channels, hidden_channels, kernel_size=3, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(max(1, hidden_channels // 4), 4 * hidden_channels)

    def forward(self, x, hcur, ccur):
        # x: (B, in_channels, H, W)
        # hcur, ccur: (B, hidden_channels, H, W)
        combined = torch.cat([x, hcur], dim=1)  # (B, in+hidden, H, W)
        conv_out = self.conv(combined)
        conv_out = self.norm(conv_out)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        cnext = f * ccur + i * g
        hnext = o * torch.tanh(cnext)
        return hnext, cnext

class NeuralMapMemory(nn.Module):
    def __init__(self, map_channels, memory_size, write_sigma=1.0, convlstm_hidden=None):
        super().__init__()
        self.map_channels = map_channels
        self.memory_size = memory_size
        self.write_sigma = write_sigma
        if convlstm_hidden is None:
            convlstm_hidden = map_channels
        self.convlstm = ConvLSTMCell(in_channels=map_channels, hidden_channels=convlstm_hidden)
        self.post_conv = nn.Sequential(
            nn.Conv2d(convlstm_hidden, map_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, map_channels // 4), map_channels),
            nn.ReLU()
        )

        # Precompute grid
        gx = torch.linspace(0, 1, memory_size)
        gy = torch.linspace(0, 1, memory_size)
        self.register_buffer('grid_x', gx.view(1, memory_size).repeat(memory_size, 1).t())  # (H,W)
        self.register_buffer('grid_y', gy.view(1, memory_size).repeat(memory_size, 1))       # (H,W)

    def encode_positions(self, positions):
        """Ensure positions are in [0,1]"""
        return torch.clamp(positions, 0.0, 1.0)

    def gaussian_mask(self, positions):
        b = positions.size(0)
        gx = self.grid_x.unsqueeze(0).expand(b, -1, -1)
        gy = self.grid_y.unsqueeze(0).expand(b, -1, -1)
        px = positions[:, 0].view(b, 1, 1)
        py = positions[:, 1].view(b, 1, 1)
        dx = (gx - px) ** 2
        dy = (gy - py) ** 2
        sigma = self.write_sigma / self.memory_size
        sigma = max(1e-3, min(sigma, 0.5)) 
        mask = torch.exp(-(dx + dy) / (2 * sigma**2))
        return mask.unsqueeze(1)  # (B,1,H,W)

    def write(self, memory, write_values, write_gate, erase_gate, positions):
        b, c, h, w = memory.shape
        
        if write_values.dim() == 1:
            write_values = write_values.view(b, c)
        
        positions = self.encode_positions(positions)
        mask = self.gaussian_mask(positions).expand(-1, c, -1, -1)  # (B, C, H, W)
        
        # FIX: Proper reshaping for broadcasting
        write_vals_map = write_values.view(b, c, 1, 1).expand(-1, -1, h, w)
        write_gate_map = write_gate.view(b, c, 1, 1).expand(-1, -1, h, w)
        erase_gate_map = erase_gate.view(b, c, 1, 1).expand(-1, -1, h, w)
        
        memory = memory * (1.0 - mask * erase_gate_map) + mask * write_gate_map * write_vals_map
        return memory

    def forward(self, memory, convlstm_state, write_values=None, write_gate=None, erase_gate=None, positions=None):
        if write_values is not None and positions is not None:
            memory = self.write(memory, write_values, write_gate, erase_gate, positions)
        h, c = convlstm_state
        hnext, cnext = self.convlstm(memory, h, c)
        memory = self.post_conv(hnext)
        return memory, (hnext, cnext)
    
class SpatialMemoryPPO(nn.Module):
    def __init__(self, input_dim, num_actions, map_channels=16, memory_size=16, hidden_size=256):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.map_channels = map_channels
        self.memory_size = memory_size
        self.hidden_size = hidden_size

        # For image inputs, use a CNN encoder instead of linear layers
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # (B,32,H/2,W/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (B,64,H/4,W/4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                          # (B,64,4,4)
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU()
        )
        enc_out = 256
        
        self.write_head = nn.Linear(enc_out, map_channels)
        self.write_gate_head = nn.Linear(enc_out, map_channels)
        self.erase_gate_head = nn.Linear(enc_out, map_channels)

        self.memory = NeuralMapMemory(map_channels=map_channels, memory_size=memory_size)
        self.reader = nn.Sequential(
            nn.Conv2d(map_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(64*4*4, hidden_size),
            nn.ReLU()
        )
        self.temporal_lstm = nn.LSTM(hidden_size + enc_out, hidden_size, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )

    def init_memory(self, batch_size=1, device='cpu'): 
        # Hidden and cell for ConvLSTM (initialized to zeros) 
        mem = torch.zeros(batch_size, self.map_channels, self.memory_size, self.memory_size, device=device) 
        h = torch.zeros(batch_size, self.map_channels, self.memory_size, self.memory_size, device=device) 
        c = torch.zeros(batch_size, self.map_channels, self.memory_size, self.memory_size, device=device) 
        return mem, (h, c)

    def init_states(self, batch_size=1, device='cpu'):
        mem, (h, c) = self.init_memory(batch_size, device)
        h_l = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c_l = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (mem, (h, c)), (h_l, c_l)

    def encode_obs(self, obs):
        return self.obs_encoder(obs)

    def read_memory(self, memory):
        return self.reader(memory)

    def get_write_params(self, obs_embed):
        """Ensure write parameters have correct dimensions"""
        write_vals = torch.tanh(self.write_head(obs_embed))  # (B, map_channels)
        write_gate = torch.sigmoid(self.write_gate_head(obs_embed))  # (B, map_channels)
        erase_gate = torch.sigmoid(self.erase_gate_head(obs_embed))  # (B, map_channels)
        return write_vals, write_gate, erase_gate

    def forward(self, obs_seq, positions=None, initial_states=None):
        """
        Forward pass for SpatialMemoryPPO.
        """
        device = obs_seq.device

        # Handle input shapes
        if obs_seq.ndim == 4:  # (B, C, H, W) - single step
            B, C, H, W = obs_seq.shape
            T = 1
            needs_sequence_dim = True
        elif obs_seq.ndim == 5:  # (B, T, C, H, W) - sequence
            B, T, C, H, W = obs_seq.shape
            needs_sequence_dim = False
        else:
            raise ValueError(f"Unsupported obs_seq shape: {obs_seq.shape}")
        
        # Handle position shapes
        if positions is not None:
            if positions.ndim == 2:  # (B, 2) - single step coordinates
                positions = positions.unsqueeze(1) if needs_sequence_dim else positions.unsqueeze(1)
            elif positions.ndim == 3:  # (B, T, 2) - sequence coordinates
                pass
            else:
                raise ValueError(f"Unsupported positions shape: {positions.shape}")

        if initial_states is None:
            (mem, convlstm_state), lstm_hidden = self.init_states(B, device)
        else:
            (mem, convlstm_state), lstm_hidden = initial_states

        logits_seq = []
        values_seq = []

        # Process each timestep
        for t in range(T):
            # Get current observation
            if needs_sequence_dim:
                obs_t = obs_seq  # (B, C, H, W)
            else:
                obs_t = obs_seq[:, t]  # (B, C, H, W)
                
            obs_emb = self.encode_obs(obs_t)  # (B, enc_out)
            
            write_vals, write_gate, erase_gate = self.get_write_params(obs_emb)
            
            # Get current position
            if positions is not None:
                if needs_sequence_dim:
                    pos = positions[:, 0]  # (B, 2)
                else:
                    pos = positions[:, t]  # (B, 2)
            else:
                pos = None
                
            # FIX: Ensure memory has correct batch size
            if mem.shape[0] != B:
                # If batch size changed, we need to adjust memory
                mem = mem[:B]  # Take first B elements
                h_conv, c_conv = convlstm_state
                h_conv = h_conv[:B]
                c_conv = c_conv[:B]
                convlstm_state = (h_conv, c_conv)
                
            mem, convlstm_state = self.memory(mem, convlstm_state,
                                            write_values=write_vals,
                                            write_gate=write_gate,
                                            erase_gate=erase_gate,
                                            positions=pos)
            
            mem_read = self.read_memory(mem)
            lstm_input = torch.cat([obs_emb, mem_read], dim=-1).unsqueeze(1)  # (B, 1, hidden_size + enc_out)
            lstm_out, lstm_hidden = self.temporal_lstm(lstm_input, lstm_hidden)
            lstm_out = lstm_out.squeeze(1)
            
            logits_seq.append(self.actor(lstm_out))
            values_seq.append(self.critic(lstm_out))

        logits_seq = torch.stack(logits_seq, dim=1)  # (B, T, A)
        values_seq = torch.stack(values_seq, dim=1)  # (B, T, 1)
        final_states = ((mem, convlstm_state), lstm_hidden)
        return logits_seq, values_seq, final_states, mem
    
class SpatialPPOAgent:
    def __init__(self, input_dim, num_actions, device='cpu',
                 lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=4,
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.1,
                 buffer_size=2048, minibatch_size=128):
        self.device = device
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size

        self.model = SpatialMemoryPPO(input_dim, num_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer, mode='max', factor=0.5, patience=200, min_lr=1e-6, threshold=0.01)


        # Initialize buffers
        self.reset_buffers()
        
        # Episode tracking
        self.episodes = []
        self.current_episode = None
        self.total_steps = 0

    def reset_buffers(self):
        """Initialize or reset experience buffers"""
        self.obs_buf = []
        self.pos_buf = []
        self.actions_buf = []
        self.logprobs_buf = []
        self.rewards_buf = []
        self.dones_buf = []
        self.values_buf = []
        self.hidden_buf = []
        self.memory_buf = []

    def start_episode(self, initial_obs, initial_position, initial_states=None):
        """Call at env.reset. initial_obs: np array D, initial_position: (2)"""
        if isinstance(initial_obs, np.ndarray):
            initial_obs = initial_obs.astype(np.float32)
        ep = {
            'obs': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'dones': [],
            'positions': [],
            'initial_states': None
        }

        if initial_states is None:
            (mem, convlstm_state), lstm_hidden = self.model.init_states(1, self.device)
            initial_states = ((mem.detach().cpu().clone(), 
                             (convlstm_state[0].detach().cpu().clone(), 
                              convlstm_state[1].detach().cpu().clone())),
                            (lstm_hidden[0].detach().cpu().clone(), 
                             lstm_hidden[1].detach().cpu().clone()))
            
        ep['initial_states'] = initial_states
        ep['obs'].append(np.array(initial_obs, dtype=np.float32))
        ep['positions'].append(np.array(initial_position, dtype=np.float32))
        self.current_episode = ep

        return ep

    def store_step(self, obs, action, logprob, reward, done, position, value, hidden_state):
        """Store experience with memory states"""
        obs_tensor = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        pos_tensor = torch.from_numpy(position.astype(np.float32)).to(self.device) if position is not None else None
        
        self.obs_buf.append(obs_tensor)
        self.pos_buf.append(pos_tensor)
        self.actions_buf.append(action)
        self.logprobs_buf.append(logprob)
        self.rewards_buf.append(reward)
        self.dones_buf.append(done)
        self.values_buf.append(value)
        
        # CRITICAL: Store the hidden state for this timestep
        self.hidden_buf.append(hidden_state)
        
        self.total_steps += 1
        
        if self.total_steps >= self.buffer_size:
            self.update()

    def remember(self, state, action, logprob, reward, done, value, hidden, memory, position_encoding, start=False, end=False):
        """
        Store a single transition in buffers for PPO updates.
        """
        # Ensure buffers exist
        if not hasattr(self, 'obs_buf'):
            self.reset_buffers()

        # FIX: Ensure the hidden state has the correct structure
        # hidden should be: ((mem, (h_conv, c_conv)), (h_lstm, c_lstm))
        if not isinstance(hidden, tuple) or len(hidden) != 2:
            raise ValueError(f"Hidden state has incorrect structure: {type(hidden)}")
        
        memory_state, lstm_state = hidden
        if not isinstance(memory_state, tuple) or len(memory_state) != 2:
            raise ValueError(f"Memory state has incorrect structure: {type(memory_state)}")
        
        if not isinstance(lstm_state, tuple) or len(lstm_state) != 2:
            raise ValueError(f"LSTM state has incorrect structure: {type(lstm_state)}")

        # Append step data (with proper CPU detach)
        self.obs_buf.append(state.cpu().detach())
        self.pos_buf.append(position_encoding.cpu().detach() if position_encoding is not None else None)
        self.actions_buf.append(action)
        self.logprobs_buf.append(logprob)
        self.rewards_buf.append(reward)
        self.dones_buf.append(done)
        self.values_buf.append(value)
        
        # Store states with proper CPU detach
        mem, convlstm_state = memory_state
        h_lstm, c_lstm = lstm_state
        h_conv, c_conv = convlstm_state
        
        hidden_cpu = (
            (
                mem.cpu().detach().clone(),
                (h_conv.cpu().detach().clone(), c_conv.cpu().detach().clone())
            ),
            (h_lstm.cpu().detach().clone(), c_lstm.cpu().detach().clone())
        )
        
        self.hidden_buf.append(hidden_cpu)
        self.memory_buf.append(hidden_cpu)  # Same as hidden for now

    def compute_gae(self, rewards, values, dones, gamma=None, lam=0.95):
        if gamma is None:
            gamma = self.gamma

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(dones[t])
            next_value = values[t + 1] if t < T - 1 else 0.0
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            gae = delta + gamma * lam * nonterminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        return advantages, returns

    def update(self, minibatch_size=None):
        """
        PPO update that preserves memory state dynamics with correct shapes
        """
        if minibatch_size is None:
            minibatch_size = self.minibatch_size
            
        if len(self.obs_buf) == 0:
            return
        
        self.model.train()
        device = self.device
        
        # Convert buffers to tensors
        obs_seq = torch.stack(self.obs_buf, dim=0).to(device)  # (T, C, H, W) - already the correct shape!
        pos_seq = torch.stack(self.pos_buf, dim=0).to(device) if self.pos_buf[0] is not None else None
        actions_t = torch.tensor(self.actions_buf, dtype=torch.long, device=device)
        old_logprobs_t = torch.tensor(self.logprobs_buf, dtype=torch.float32, device=device)
        rewards = np.array(self.rewards_buf, dtype=np.float32)
        dones = np.array(self.dones_buf, dtype=np.bool_)
        values = np.array(self.values_buf, dtype=np.float32)

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        
        if advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Training loop
        for epoch in range(self.epochs):
            T = len(obs_seq)
            perm = np.random.permutation(T)

            for start_idx in range(0, T, minibatch_size):
                end_idx = min(start_idx + minibatch_size, T)
                idx = perm[start_idx:end_idx]
                current_batch_size = len(idx)

                # Use the stored memory states
                if hasattr(self, 'hidden_buf') and len(self.hidden_buf) > idx[0]:
                    init_states = self.hidden_buf[idx[0]]
                    
                    # Unpack and move to device
                    memory_state, lstm_state = init_states
                    mem, convlstm_state = memory_state
                    h_conv, c_conv = convlstm_state
                    h_lstm, c_lstm = lstm_state
                    
                    # Ensure batch size matches
                    if mem.shape[0] != current_batch_size:
                        mem = mem[:1].expand(current_batch_size, -1, -1, -1)
                        h_conv = h_conv[:1].expand(current_batch_size, -1, -1, -1)
                        c_conv = c_conv[:1].expand(current_batch_size, -1, -1, -1)
                        h_lstm = h_lstm[:, :1, :].expand(-1, current_batch_size, -1)
                        c_lstm = c_lstm[:, :1, :].expand(-1, current_batch_size, -1)
                    
                    # Move to device
                    mem = mem.to(device)
                    h_conv = h_conv.to(device)
                    c_conv = c_conv.to(device)
                    h_lstm = h_lstm.to(device)
                    c_lstm = c_lstm.to(device)
                    
                    init_states = ((mem, (h_conv, c_conv)), (h_lstm, c_lstm))
                else:
                    # Fallback: use zero states
                    init_states = self.model.init_states(current_batch_size, device)

                # Pass as single steps (B, C, H, W)
                logits_seq, values_seq, final_states, _ = self.model(
                    obs_seq=obs_seq[idx],  # (minibatch_size, C, H, W) - single step
                    positions=pos_seq[idx] if pos_seq is not None else None,  # (minibatch_size, 2)
                    initial_states=init_states
                )
                
                # Handle output dimensions - your model returns sequences even for single steps
                if logits_seq.dim() == 3:  # (B, T, A) where T=1 for single steps
                    logits_seq = logits_seq.squeeze(1)  # (B, A)
                if values_seq.dim() == 3:  # (B, T, 1) where T=1 for single steps
                    values_seq = values_seq.squeeze(1).squeeze(-1)  # (B,)

                # PPO Loss
                dist = torch.distributions.Categorical(logits=logits_seq)
                new_logprobs = dist.log_prob(actions_t[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - old_logprobs_t[idx])
                surr1 = ratio * advantages_t[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_t[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * F.mse_loss(values_seq, returns_t[idx])
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                # Optimization
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient monitoring
                total_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        total_norm += param.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

        success_rate = 0
        if hasattr(self, 'episodes') and self.episodes:
            # Use the last few episodes to calculate success rate
            recent_episodes = self.episodes[-10:] if len(self.episodes) >= 10 else self.episodes
            success_rate = np.mean([1.0 if any(ep['dones']) else 0.0 for ep in recent_episodes])
        else:
            success_rate = np.mean(self.dones_buf) if len(self.dones_buf) > 0 else 0
            
        # Clear buffers
        self.reset_buffers()
        self.total_steps = 0

        # Scheduler step
        self.scheduler.step(success_rate)
        
    def act(self, state, memory=None, position=None):
        self.model.eval()
        device = next(self.model.parameters()).device

        # --- Ensure batch dimension ---
        if state.ndim == 3:  # (C,H,W)
            state = state.unsqueeze(0)  # (1,C,H,W)
        elif state.ndim != 4:
            raise ValueError(f"Unsupported state shape: {state.shape}")

        if position is not None:
            if position.ndim == 1:  # (2,) - single coordinate
                position = position.unsqueeze(0)  # (1,2)
            elif position.ndim == 2:  # (B,2) - coordinates
                pass
            else:
                raise ValueError(f"Unsupported position shape: {position.shape}")

        if memory is None:
            memory = self.model.init_states(batch_size=state.shape[0], device=device)

        # --- Forward pass ---
        with torch.no_grad():
            logits, values, final_states, _ = self.model(
                obs_seq=state.to(device),  # (1, C, H, W)
                positions=position.to(device) if position is not None else None,  # (1, 2)
                initial_states=memory
            )
            # Handle single-step output
            if logits.dim() == 3 and logits.shape[1] == 1:  # (B, 1, A)
                logits = logits.squeeze(1)  # (B, A)
            if values.dim() == 3 and values.shape[1] == 1:  # (B, 1, 1)
                values = values.squeeze(1).squeeze(-1)  # (B,)
                
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)

        return action.item(), logprob.item(), values.item(), final_states
    
    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])