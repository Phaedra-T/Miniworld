
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class LSTMAgentModel(nn.Module):
    def __init__(self, num_actions, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        
        # CNN Feature Encoder (Same as AgentModelRGB)
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256), # Output 256 features
            nn.ReLU(),
        )
        
        # Recurrent LSTM memory module
        self.lstm = nn.LSTM(256, hidden_size, batch_first=True)

        # Separate actor-critic heads
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
         
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.hidden_size, device=device),
                torch.zeros(1, batch_size, self.hidden_size, device=device))    

    def forward(self, obs, hidden):
        # obs shape: (B, T, C, H, W) or (B, C, H, W) if T=1 handled carefully
        if obs.dim() == 4:
            obs = obs.unsqueeze(1) # (B, 1, C, H, W)
            
        B, T, C, H, W = obs.shape
        
        # Flatten batch and time for CNN
        obs_flat = obs.view(B * T, C, H, W)
        
        # Normalize if uint8
        if obs_flat.max() > 1.0:
            obs_flat = obs_flat.float() / 255.0
            
        features = self.obs_encoder(obs_flat) # (B*T, 256)
        features = features.view(B, T, -1)
        
        lstm_out, hidden = self.lstm(features, hidden)
        
        # Process all timesteps
        logits = self.actor(lstm_out)
        values = self.critic(lstm_out)
        
        return logits, values, hidden

class LSTMAgent:
    def __init__(self, action_dim, device, lr=1e-4, gamma=0.99, 
                 lam=0.95, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5, n_steps=128):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        
        self.model = LSTMAgentModel(action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-8)
        
        self.reset_buffers()

    def reset_buffers(self):
        self.obs_buf = []
        self.actions_buf = []
        self.rewards_buf = []
        self.values_buf = []
        self.done_buf = []
        # Store initial hidden state for the sequence
        self.initial_hidden = None 

    def act(self, obs, hidden):
        # obs: (H, W, C) numpy -> (1, 1, C, H, W) tensor
        if not isinstance(obs, torch.Tensor):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_t = obs.to(self.device)
            
        # Reshape to (B, T, C, H, W) -> (1, 1, 3, H, W)
        obs_t = obs_t.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        
        # Capture the hidden state at the start of the n_steps rollout
        if self.initial_hidden is None:
            self.initial_hidden = (hidden[0].detach(), hidden[1].detach())

        with torch.no_grad():
            logits, value, next_hidden = self.model(obs_t, hidden)
        
        dist = Categorical(logits=logits.squeeze())
        action = dist.sample()
        logp = dist.log_prob(action)

        return action.item(), logp, value.item(), next_hidden

    def store(self, obs, action, reward, done, value):
        # We store frames as CPU tensors to save GPU memory, similar to other agents
        # obs is numpy (H, W, C). Store as (C, H, W) float/byte to save space?
        # Let's just store as tensor.
        obs_t = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1) # (C, H, W)
        self.obs_buf.append(obs_t)
        self.actions_buf.append(action)
        self.rewards_buf.append(reward)
        self.done_buf.append(done)
        self.values_buf.append(value)

    def compute_gae(self, last_value):
        values = np.array(self.values_buf + [last_value], dtype=np.float32)
        rewards = np.array(self.rewards_buf, dtype=np.float32)
        dones = np.array(self.done_buf, dtype=np.float32)

        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t+1] * nonterminal - values[t]
            lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
            adv[t] = lastgaelam

        returns = adv + values[:-1]
        return torch.tensor(adv, device=self.device), torch.tensor(returns, device=self.device)

    def update(self, last_hidden, done=False, force=False):
        if len(self.obs_buf) == 0:
            self.initial_hidden = None
            return None

        if len(self.obs_buf) < self.n_steps and not force:
            return None

        # Calculate last value for GAE bootstrap
        if done:
            last_val = 0.0
        else:
            # Last obs needs processing
            last_obs = self.obs_buf[-1].unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, C, H, W)
            with torch.no_grad():
                _, last_val, _ = self.model(last_obs, last_hidden)
            last_val = last_val.item()
        
        adv, returns = self.compute_gae(last_val)
        # Normalize advantages safely for very short rollouts
        if adv.numel() <= 1:
            adv = adv * 0.0
        else:
            adv_std = adv.std(unbiased=False)
            if adv_std < 1e-6:
                adv = adv - adv.mean()
            else:
                adv = (adv - adv.mean()) / (adv_std + 1e-8)

        # Prepare sequence
        # obs_buf has (T, C, H, W) eventually
        obs_seq = torch.stack(self.obs_buf).unsqueeze(0).to(self.device) # (1, T, C, H, W)
        actions = torch.tensor(self.actions_buf, device=self.device)

        # Re-forward through the model using the initial hidden state
        logits, values, _ = self.model(obs_seq, self.initial_hidden)
        
        logits = logits.squeeze(0)
        values = values.squeeze()

        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(logp * adv.detach()).mean()
        value_loss = (values - returns).pow(2).mean()
        
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.reset_buffers()
        return {
            "loss": loss.item(), 
            "entropy": entropy.item(), 
            "p_loss": policy_loss.item(), 
            "v_loss": value_loss.item()
        }

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
