import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class MultiScaleEncoder(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_size=128):  # Reduced for stability
        super().__init__()
        self.hidden_size = hidden_size
        
        # Simpler feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_size),
            nn.ReLU(),
        )
        
        # LSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        
        # Heads
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x, hidden=None):
        # Input: (batch, seq, features) or (batch, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, D) -> (B, 1, D)
        
        batch_size, seq_len, feat_dim = x.size()
        
        # Extract features
        x_flat = x.reshape(batch_size * seq_len, feat_dim)
        features = self.feature_net(x_flat)
        features = features.reshape(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, hidden = self.lstm(features, hidden)
        
        # Use ALL timesteps for training, last for acting
        if self.training and seq_len > 1:
            # For training: process entire sequence
            logits = self.actor(lstm_out.reshape(batch_size * seq_len, -1))
            value = self.critic(lstm_out.reshape(batch_size * seq_len, -1))
            logits = logits.reshape(batch_size, seq_len, -1)
            value = value.reshape(batch_size, seq_len, -1)
        else:
            # For acting: use last timestep only
            last_out = lstm_out[:, -1, :]
            logits = self.actor(last_out)
            value = self.critic(last_out)
        
        return logits, value, hidden
    
    def init_hidden(self, batch_size=1, device="cpu"):
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h, c)

class PPOAgent:
    def __init__(self, input_dim, num_actions, device, lr=3e-4, gamma=0.99,  
                 clip_eps=0.2, epochs=4, entropy_coef=0.03, value_coef = 0.5, max_grad_norm=0.5,
                 buffer_size=2048):  # Better defaults
        self.device = device
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size

        self.model = MultiScaleEncoder(input_dim, num_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        
        # Buffers
        self.obs_buf = []
        self.action_buf = []
        self.logprob_buf = []
        self.reward_buf = []
        self.done_buf = []
        self.value_buf = []
        self.hidden_buf = []
        self.episode_starts = []
        
        self.clear_memory()

    def clear_memory(self):
        self.obs_buf.clear()
        self.action_buf.clear()
        self.logprob_buf.clear()
        self.reward_buf.clear()
        self.done_buf.clear()
        self.value_buf.clear()
        self.hidden_buf.clear()
        self.episode_starts = []

    def act(self, state, hidden):
        """Action selection for single step"""
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        # Ensure proper dimensions for single step
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (D) -> (1, D)
        
        # Set model to eval mode for acting
        self.model.eval()
        with torch.no_grad():
            logits, value, new_hidden = self.model(state, hidden)
            
            # Handle output dimensions
            if logits.dim() == 2 and logits.size(0) == 1:
                logits = logits.squeeze(0)
            if value.dim() == 2 and value.size(0) == 1:
                value = value.squeeze(0).squeeze(-1)
            
            probs = torch.distributions.Categorical(logits=logits)
            action = probs.sample()
            log_prob = probs.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item()), new_hidden
    
    def remember(self, obs, action, logprob, reward, done, value, hidden, start_of_episode=False):
        """Store experience."""
        if len(self.obs_buf) >= self.buffer_size:
            return

        if isinstance(obs, np.ndarray):
            self.obs_buf.append(obs.copy())
        else:
            self.obs_buf.append(np.array(obs, dtype=np.float32))

        self.action_buf.append(int(action))
        self.logprob_buf.append(float(logprob))
        self.reward_buf.append(float(reward))
        self.done_buf.append(bool(done))
        self.value_buf.append(float(value))

        # store hidden *per episode* when flagged
        if start_of_episode:
            # store deep copies of hidden states aligned to episode starting index
            self.hidden_buf.append((hidden[0].detach().cpu().clone(), hidden[1].detach().cpu().clone()))
            # also store the buffer index where this episode started
            if not hasattr(self, 'episode_starts'):
                self.episode_starts = []
            self.episode_starts.append(len(self.obs_buf)-1)

    def compute_advantages(self, rewards, values, dones, gamma=None, lam=0.95):
        """Compute Generalized Advantage Estimation (GAE) for PPO.
        
        Args:
            rewards (list or np.ndarray): Rewards for each timestep.
            values (list or np.ndarray): Value predictions for each timestep.
            dones (list or np.ndarray): Done flags for each timestep.
            gamma (float): Discount factor. Uses self.gamma if None.
            lam (float): GAE lambda parameter (typically 0.95).
        
        Returns:
            advantages (np.ndarray): Advantage estimates for each timestep.
            returns (np.ndarray): Target value estimates (advantages + values)."""

        if gamma is None:
            gamma = self.gamma

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        gae = 0.0
        next_value = 0.0

        # Iterate backward through the episode
        for t in reversed(range(T)):
            # Is next state terminal? (0 if done, else 1)
            next_nonterminal = 1.0 - float(dones[t])

            # For last timestep, next_value = 0 if episode ended, else value[t+1]
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            # Temporal-difference residual
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]

            # GAE recursive formula
            gae = delta + gamma * lam * next_nonterminal * gae
            advantages[t] = gae

            # Compute return = value + advantage
            returns[t] = advantages[t] + values[t]

        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns


    def update(self):
        """PPO update with proper sequence processing"""
        if len(self.obs_buf) < 256:  # Smaller minibatch
            return
         
        # Set model to training mode
        self.model.train()

        # convert buffers to tensors
        obs_arr = np.array(self.obs_buf, dtype=np.float32)
        actions_t = torch.tensor(self.action_buf, dtype=torch.long, device=self.device)
        old_logprobs_t = torch.tensor(self.logprob_buf, dtype=torch.float32, device=self.device)
        rewards = np.array(self.reward_buf, dtype=np.float32)
        dones = np.array(self.done_buf, dtype=np.bool_)
        values_arr = np.array(self.value_buf, dtype=np.float32)

        # find episode start indices (self.episode_starts)
        starts = getattr(self, 'episode_starts', [0])
        starts = starts + [len(obs_arr)]  # sentinel end
        # iterate episodes
        all_indices = []
        for i in range(len(starts)-1):
            s = starts[i]
            e = starts[i+1]
            # slice episode
            ep_obs = torch.tensor(obs_arr[s:e], dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, seq, feat)
            ep_actions = actions_t[s:e]
            ep_old_logprobs = old_logprobs_t[s:e]
            ep_rewards = rewards[s:e].tolist()
            ep_values = values_arr[s:e].tolist()
            ep_dones = dones[s:e].tolist()

            # compute advs, returns 
            advs, returns = self.compute_advantages(ep_rewards, ep_values, ep_dones)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
            advs_t = torch.tensor(advs, dtype=torch.float32, device=self.device)
            if advs_t.numel() > 1:
                advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)
            # Clamp advantages to avoid exploding ratios
            advs_t = torch.clamp(advs_t, -10, 10)

            initial_hidden = (self.hidden_buf[i][0].to(self.device), self.hidden_buf[i][1].to(self.device))
            # forward once, get logits & values for all timesteps
            logits, values, _ = self.model(ep_obs, initial_hidden)
            logits = logits.squeeze(0)
            values = values.view(-1)
            
            # Compute losses
            probs = torch.distributions.Categorical(logits=logits)
            new_logprobs = probs.log_prob(ep_actions)
            entropy = probs.entropy().mean()
            
            # PPO clipped objective
            ratio = torch.exp(new_logprobs - ep_old_logprobs)
            surr1 = ratio * advs_t
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs_t
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * F.mse_loss(values, returns_t)
            
            # Simplified loss
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            if torch.isnan(loss):
                print("⚠️ NaN detected in loss, skipping update")
                self.clear_memory()
                return
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # Clear memory after update
        self.clear_memory()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))