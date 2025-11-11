import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CuriosityModule:
    """
    Predicts next state given (state, action) and produces an intrinsic
    reward proportional to prediction error. Uses one-hot action encoding
    for discrete actions and stable normalization.
    """
    def __init__(self, state_dim, num_actions, hidden_dim=128,
                 device="cpu", learning_rate=1e-4, scale=0.01):
        self.device = device
        self.num_actions = num_actions
        self.scale = scale

        # Forward model: (state + onehot_action) → next_state
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        ).to(device)

        self.optimizer = optim.Adam(self.forward_model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Running stats for normalization
        self.running_mean = 0.0
        self.running_var = 1.0
        self.beta = 0.99  # for exponential moving average


    def compute_intrinsic_reward(self, state, action, next_state):
        # Convert to tensors
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        ns = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # One-hot encode action
        a = torch.zeros((1, self.num_actions), device=self.device)
        a[0, int(action)] = 1.0

        inp = torch.cat([s, a], dim=1)
        pred = self.forward_model(inp)

        # Prediction error
        loss = self.loss_fn(pred, ns)

        # Train forward model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), 1.0)
        self.optimizer.step()

        # Normalize error (EMA of mean/var)
        err = loss.item()
        self.running_mean = self.beta * self.running_mean + (1 - self.beta) * err
        self.running_var = self.beta * self.running_var + (1 - self.beta) * (err - self.running_mean) ** 2
        std = np.sqrt(self.running_var) + 1e-8

        normalized = max(0.0, (err - self.running_mean) / std)
        return float(self.scale * normalized)  # scaled intrinsic reward


class NoCuriosity:
    """Dummy curiosity module (for ablation/testing)."""
    def compute_intrinsic_reward(self, *args, **kwargs):
        return 0.0
