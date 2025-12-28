#main_spat.py
import os
import torch
import gymnasium as gym
import miniworld.envs
import matplotlib.pyplot as plt
from spat_ppo_agent import PPOAgent
from utils import build_spat_state, get_latest_checkpoint, train_spat, log_metrics, plot_training_curve


# --- Hyperparameters ---
env_name = "MiniWorld-ColorCueCorridor-v0" 
NUM_EPISODES = 10000 
LR = 7e-5 
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP = 0.15
ENTROPY = 0.001
VALUE_COEF = 0.25
MAX_GRAD_NORM = 0.25
ROLLOUT_LEN = 512 
EPOCHS = 4 
CHUNK_LEN = 32
WRITE_WARMUP_STEPS = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = os.path.join(script_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# --- Initialize environment ---
env = gym.make(env_name, render_mode= None) #"rgb_array" )
env.set_wrapper_attr(name="max_episode_steps", value=500)
obs, _ = env.reset()

NORM_SCALE = max(
    env.unwrapped.max_x - env.unwrapped.min_x,
    env.unwrapped.max_z - env.unwrapped.min_z
)

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
    chunk_len=CHUNK_LEN,                  
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
results = train_spat(env, NUM_EPISODES, agent, DEVICE, ROLLOUT_LEN, goal_pos)
#results = train_with_curriculum(env_fns=env_factories,env_names=env_names,start_pool_size=1,num_episodes=5000,agent=agent,device=DEVICE,rollout=ROLLOUT_LEN,add_threshold=0.65,rolling_window=200,require_per_env=False, max_envs_to_add=4)
# --- Save trained agent ---
new_checkpoint = os.path.join(
    checkpoint_dir,
    f"ppo_{env.spec.id.lower().replace('-', '_')}.pt"
)
#agent.save(new_checkpoint)

# --- Logging ---
df = log_metrics(results, env, checkpoint_dir)
plot_training_curve(df, env)

env.close()
