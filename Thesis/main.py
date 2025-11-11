import os
import torch
import gymnasium as gym
import miniworld.envs
import matplotlib.pyplot as plt
from ppo_lstm_agent import PPOAgent  
#from ppo_agent import PPOAgent 
from utils import build_state, get_latest_checkpoint, train, log_metrics,plot_training_curve
#from debug import debug_exploration

# --- Hyperparameters ---
env_name = "MiniWorld-StaticMaze-v0"
NUM_EPISODES = 5000
LR = 1e-4
GAMMA = 0.99
ROLLOUT_LEN = 256
EPOCHS = 4
NORM_SCALE = 12.0 
#DEVICE = torch.device("cpu") #"mps" if torch.backends.mps.is_available() else 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = os.path.join(script_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# --- Initialize environemnt ---
env = gym.make(env_name, render_mode=None)
env.set_wrapper_attr(name="max_episode_steps", value=800)
obs, _ = env.reset()

#--- Build Initial State ---
agent_spawn = env.unwrapped.agent.pos
goal_pos = env.unwrapped.box.pos
state = build_state(agent_spawn, goal_pos, env)
input_dim = len(state)
num_actions = env.action_space.n

#--- Initialise agent ---
agent = PPOAgent(input_dim, num_actions, DEVICE, lr=LR, entropy_coef=0.05)
    
#--- Load latest checkpoint ---
latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    agent.load(latest_checkpoint)
    print(f"✅ Loaded checkpoint: {latest_checkpoint}")
else:
    print("⚠️ No checkpoint found. Training from scratch!")

# --- Training ---    
success_log = train(env, NUM_EPISODES, agent, DEVICE, ROLLOUT_LEN)

# --- Save Trained Agent ---
new_checkpoint = os.path.join(checkpoint_dir, f"ppo_{env.spec.id.lower().replace('-','_')}.pt")
agent.save(new_checkpoint)

df = log_metrics(success_log,env,checkpoint_dir)
plot_training_curve(df, env)
env.close()
