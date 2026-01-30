
import os
import torch
import gymnasium as gym
import miniworld.envs
import matplotlib.pyplot as plt
from Thesis.lstm.agent import LSTMAgent
from Thesis.lstm.utils import get_latest_checkpoint, trainA2C_RGB, log_metrics, plot_training_curve

# --- Hyperparameters ---
ENV_NAME = "MiniWorld-MazeS3-v0"
NUM_EPISODES = 8000
LR = 2.5e-4
GAMMA = 0.99
ROLLOUT_LEN = 128
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 LSTM RGB Training on {ENV_NAME}")
print(f"Device: {DEVICE}")

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = os.path.join(script_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# --- Initialize environment ---
env = gym.make(ENV_NAME, render_mode=None)
env.set_wrapper_attr(name="max_episode_steps", value=400) # Match RGB agent setting
obs, _ = env.reset()

action_dim = env.action_space.n

#--- Initialize Agent ---
agent = LSTMAgent(
    action_dim=action_dim, 
    device=DEVICE, 
    lr=LR, 
    gamma=GAMMA, 
    entropy_coef=ENTROPY_COEF, 
    value_coef=VALUE_COEF, 
    n_steps=ROLLOUT_LEN
)
    
#--- Load latest checkpoint ---
latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    agent.load(latest_checkpoint)
    print(f"✅ Loaded checkpoint: {latest_checkpoint}")
else:
    print("⚠️ No checkpoint found. Training from scratch!")

# --- Training ---    
success_log = trainA2C_RGB(env, NUM_EPISODES, agent, DEVICE, ROLLOUT_LEN)

# --- Save Trained Agent ---
new_checkpoint = os.path.join(checkpoint_dir, f"final_{env.spec.id.lower().replace('-','_')}_lstm_rgb.pt")
agent.save(new_checkpoint)

df = log_metrics(success_log, env, checkpoint_dir)
plot_training_curve(df, env)
env.close()
