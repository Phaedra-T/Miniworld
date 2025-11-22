#main_spat.py
import os
import torch
import gymnasium as gym
import miniworld.envs
import matplotlib.pyplot as plt
from spat_ppo_agent import PPOAgent
from utils import build_spat_state, get_latest_checkpoint, train_spat, log_metrics, plot_training_curve


# --- Hyperparameters ---
env_name = "MiniWorld-OneRoom-v0"
NUM_EPISODES = 2000
LR = 3e-4
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP = 0.2
ENTROPY = 0.1
VALUE_COEF = 0.5
MAX_GRAD_NORM = 1.0
ROLLOUT_LEN = 512
EPOCHS = 4
MINIBATCHES = 4
WRITE_WARMUP_STEPS = 800

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = os.path.join(script_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# --- Initialize environment ---
env = gym.make(env_name, render_mode= None)
env.set_wrapper_attr(name="max_episode_steps", value=800)
obs, _ = env.reset()

# --- Build Initial State ---
agent_spawn = env.unwrapped.agent.pos
goal_pos = env.unwrapped.box.pos
state = build_spat_state(agent_spawn, goal_pos, env)
input_dim = state.shape
num_actions = env.action_space.n

print(f"State dimension: {input_dim}")
print(f"Number of actions: {num_actions}")

# --- Initialize agent correctly ---
agent = PPOAgent(
    obs_shape=input_dim,          
    action_dim=num_actions,
    device=DEVICE,
    gamma=GAMMA,
    lam=GAE_LAMBDA,
    clip_ratio=CLIP,             
    lr=LR,
    max_grad_norm=MAX_GRAD_NORM,
    rollout_len=ROLLOUT_LEN,
    chunk_len=64,                  
    entropy_coef=ENTROPY,
    value_coef=VALUE_COEF,
    epochs=EPOCHS,
    write_warmup_steps=WRITE_WARMUP_STEPS
)

# --- Load latest checkpoint ---
latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    agent.load(latest_checkpoint)
    print(f"✅ Loaded checkpoint: {latest_checkpoint}")
else:
    print("⚠️ No checkpoint found. Training from scratch!")

# --- Training ---
success_log = train_spat(env, NUM_EPISODES, agent, DEVICE, ROLLOUT_LEN, agent_spawn, goal_pos)

# --- Save trained agent ---
new_checkpoint = os.path.join(
    checkpoint_dir,
    f"ppo_{env.spec.id.lower().replace('-', '_')}.pt"
)
agent.save(new_checkpoint)

# --- Logging ---
df = log_metrics(success_log, env, checkpoint_dir)
plot_training_curve(df, env)

env.close()
