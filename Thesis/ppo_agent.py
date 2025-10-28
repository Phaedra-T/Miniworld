import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ===================================
# PPO Neural Network Architecture
# ===================================
class MLP_PPO(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)


# ===================================
# PPO Agent Class
# ===================================
class PPOAgent:
    def __init__(self, input_dim, num_actions, device, lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=4):
        self.device = device
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs

        self.model = MLP_PPO(input_dim, num_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # PPO buffers
        self.obs_buf, self.action_buf, self.logprob_buf = [], [], []
        self.reward_buf, self.done_buf, self.value_buf = [], [], []

    # ----------------------------
    # Action selection
    # ----------------------------
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, value = self.model(state)
        probs = torch.distributions.Categorical(logits=logits)
        action = probs.sample()
        return action.item(), probs.log_prob(action).item(), value.item()

    # ----------------------------
    # Memory management
    # ----------------------------
    def remember(self, obs, action, logprob, reward, done, value):
        self.obs_buf.append(obs)
        self.action_buf.append(action)
        self.logprob_buf.append(logprob)
        self.reward_buf.append(reward)
        self.done_buf.append(done)
        self.value_buf.append(value)

    def clear_memory(self):
        self.obs_buf, self.action_buf, self.logprob_buf = [], [], []
        self.reward_buf, self.done_buf, self.value_buf = [], [], []

    # ----------------------------
    # PPO update
    # ----------------------------
    def update(self):
        # Compute returns and advantages
        rewards, dones, values = np.array(self.reward_buf), np.array(self.done_buf), np.array(self.value_buf)
        returns, advs = [], []
        gae, next_value = 0, 0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * 0.95 * (1 - dones[i]) * gae
            advs.insert(0, gae)
            returns.insert(0, gae + values[i])
            next_value = values[i]

        obs = torch.FloatTensor(self.obs_buf).to(self.device)
        actions = torch.LongTensor(self.action_buf).to(self.device)
        old_logprobs = torch.FloatTensor(self.logprob_buf).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advs = torch.FloatTensor(advs).to(self.device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # PPO optimization
        for _ in range(self.epochs):
            logits, values = self.model(obs)
            probs = torch.distributions.Categorical(logits=logits)
            entropy = probs.entropy().mean()
            new_logprobs = probs.log_prob(actions)

            ratio = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - values.squeeze()).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        self.clear_memory()

    # ----------------------------
    # Layer freezing
    # ----------------------------
    def freeze_shared(self, freeze=True, partial=False):
        layers = list(self.model.shared.children())
        if partial:
            to_freeze = layers[:2]  # freeze first two layers only
        else:
            to_freeze = layers
        for layer in to_freeze:
            for param in layer.parameters():
                param.requires_grad = not freeze

    # ----------------------------
    # Model saving/loading
    # ----------------------------
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
