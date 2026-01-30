# main_spat_rgb.py
import os
import torch
import gymnasium as gym
import miniworld.envs
import matplotlib.pyplot as plt

from agent import NeuralMapAgent
from utils import get_latest_checkpoint, train_nm, log_metrics, plot_training_curve

# --- Configuration ---
ENV_NAME = "MiniWorld-OneRoomS6-v0"
LR = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY = 0.02
VALUE_COEF = 0.5
MAX_GRAD_NORM = 5.0
N_STEPS = 128
WRITE_WARMUP_STEPS = 5000
NUM_EPISODES = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 RGB Neural Map Training on {ENV_NAME}")

# --- Environment Setup ---
env = gym.make(ENV_NAME, render_mode=None)
env.set_wrapper_attr(name="max_episode_steps", value=200)
obs, _ = env.reset()
goal_pos = env.unwrapped.box.pos
num_actions = env.action_space.n
env_size = env.unwrapped.max_x - env.unwrapped.min_x

agent = NeuralMapAgent(
    env_size=env_size,
    action_dim=num_actions,
    device=DEVICE,
    gamma=GAMMA,
    lam=GAE_LAMBDA,
    lr=LR,
    max_grad_norm=MAX_GRAD_NORM,
    n_steps=N_STEPS,
    entropy_coef=ENTROPY,
    value_coef=VALUE_COEF,
    write_warmup_steps=WRITE_WARMUP_STEPS,
)

checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    agent.load(latest_checkpoint)
    print(f"✅ Loaded checkpoint: {latest_checkpoint}")
else:
    print("⚠️ No checkpoint found. Training from scratch!")

print("Starting training...")
results = train_nm(env, NUM_EPISODES, agent, DEVICE, N_STEPS, goal_pos, checkpoint_dir)

new_checkpoint = os.path.join(checkpoint_dir, f"final_{env.spec.id.lower().replace('-','_')}_256.pt")
agent.save(new_checkpoint)

df = log_metrics(results, env, checkpoint_dir)
plot_training_curve(df, env)
env.close()
