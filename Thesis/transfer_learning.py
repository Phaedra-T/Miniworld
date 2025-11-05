import os
import torch
import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import miniworld.envs
from ppo_agent import PPOAgent  # import your PPOAgent class
# --- Hyperparameters ---
LR = 1e-4
GAMMA = 0.99
ROLLOUT_LEN = 256
EPOCHS = 4
NUM_EPISODES = 5000
DEVICE = torch.device("cpu") #"mps" if torch.backends.mps.is_available() else 

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = os.path.join(script_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# --- Auto-detect latest checkpoint ---
def get_latest_checkpoint(path):
    files = [f for f in os.listdir(path) if f.endswith(".pt")]
    if not files:
        return None
    files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(path, f)), reverse=True)
    return os.path.join(path, files[0])

latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

# --- Global normalization scale ---
NORM_SCALE = 12.0  # the largest environment dimension you'll train on (e.g., hallway size 12)

# --- New environment ---
env = gym.make("MiniWorld-StaticMazeSmall-v0",  render_mode="rgb_array")
env.set_wrapper_attr(name="max_episode_steps", value=500)
obs, _ = env.reset()

#plt.ion()
#fig, (ax_top, ax_fp) = plt.subplots(1, 2, figsize=(8, 4))

room = env.unwrapped.rooms[0] 
room_width = room.max_x - room.min_x
room_depth = room.max_z - room.min_z
max_dist = np.sqrt(room_width**2 + room_depth**2)
agent_pos = env.unwrapped.agent.pos
goal_pos = env.unwrapped.box.pos

def build_state(agent_pos, goal_pos, env): 
    # agent yaw 
     yaw = env.unwrapped.agent.dir 
     rel_x = (goal_pos[0] - agent_pos[0]) / NORM_SCALE 
     rel_z = (goal_pos[2] - agent_pos[2]) / NORM_SCALE 
     return np.array([ agent_pos[0] / NORM_SCALE, 
                      agent_pos[2] / NORM_SCALE, goal_pos[0] / NORM_SCALE, 
                      goal_pos[2] / NORM_SCALE, np.cos(yaw), np.sin(yaw), 
                      rel_x, rel_z ], dtype=np.float32)

state = build_state(agent_pos, goal_pos, env)
input_dim = len(state)
num_actions = env.action_space.n

# --- Initialize agent ---
agent = PPOAgent(input_dim, num_actions, DEVICE, lr=LR, entropy_coef= 0.05)
agent.freeze_shared(True)  # freeze early for adaptation

# --- Load pretrained weights safely ---
if latest_checkpoint:
    checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)
    model_dict = agent.model.state_dict()
    

    # Filter compatible keys
    pretrained_dict = {
        k: v for k, v in checkpoint.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }

    missing_keys = set(model_dict.keys()) - set(pretrained_dict.keys())
    model_dict.update(pretrained_dict)
    agent.model.load_state_dict(model_dict)

    print(f"✅ Loaded {len(pretrained_dict)} layers from {latest_checkpoint}")
    print(f"⚪ Skipped {len(missing_keys)} unmatched keys")
else:
    print("⚠️ No checkpoint found. Training from scratch!")


# --- Attach StepLR scheduler for LR decay ---
agent.scheduler = torch.optim.lr_scheduler.StepLR(agent.optimizer, step_size=200, gamma=0.5)

# --- Training ---
success_count = 0
success_log = []
episode_returns = []
def pos_key(pos): 
    return (round(pos[0], 1), round(pos[2], 1))

#agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=3e-4)
#agent.entropy_coef = 0.02  # up from 0.01

for episode in range(NUM_EPISODES):
    hidden = agent.model.init_hidden(batch_size=1, device=DEVICE)
    success = False
    obs = env.reset()[0]

    agent_pos = env.unwrapped.agent.pos
    goal_pos = env.unwrapped.box.pos
    positions = []  # reset path tracker
    visited = set()

    state = build_state(agent_pos, goal_pos, env)

    done = False
    episode_reward = 0
    steps_in_episode = 0
          
    while not done:
        # ---- Agent action ----
        action, log_prob, value, hidden = agent.act(state, hidden)
        prev_pos = env.unwrapped.agent.pos.copy()
        positions.append((env.unwrapped.agent.pos[0], env.unwrapped.agent.pos[2]))

        next_obs, _, terminated, truncated, _ = env.step(action)
        # Keep memory content, drop autograd history
        hidden = (hidden[0].detach(), hidden[1].detach())
        done = terminated or truncated
        curr_pos = env.unwrapped.agent.pos.copy()

       # ---- Reward shaping ----
        old_dist = np.linalg.norm(prev_pos - goal_pos) 
        new_dist = np.linalg.norm(curr_pos - goal_pos) 
        delta = (old_dist - new_dist)/NORM_SCALE # positive if moved closer 
        
        reward = 0.5 * delta - 0.01 
        if pos_key(curr_pos) not in visited: 
            reward += 0.01 # novelty bonus 
            visited.add(pos_key(curr_pos)) 
            
        if new_dist < 0.4: 
            reward += 10.0 
            done = True 
            terminated = True 

        # clip reward to keep gradient stable 
        reward = float(np.clip(reward, -1.0, 1.0))
        next_state = build_state(curr_pos, goal_pos, env)

        agent.remember(state, action, log_prob, reward, done, value)
        hidden = (hidden[0].detach(), hidden[1].detach())
        state = next_state
        episode_reward += reward
        steps_in_episode += 1

        # ---- PPO update when buffer full ----
        if len(agent.obs_buf) >= ROLLOUT_LEN:
            agent.update()
            #agent.scheduler.step()  # ✅ decay LR periodically

        #env.render()

        # --- Custom render composite ---
       # top_view = env.unwrapped.render_top_view()   # top-down RGB array
       #fp_view = env.render()       # first-person RGB array

        #ax_top.clear()
        #ax_fp.clear()
        #ax_top.imshow(top_view)
        #ax_fp.imshow(fp_view)
        #ax_top.set_title("Top-Down View")
        #ax_fp.set_title("Agent View")
        #ax_top.axis('off')
        #ax_fp.axis('off')
        #plt.pause(0.001)

    # Gradual unfreezing
    if episode == 500:
        agent.freeze_shared(False, partial=True)
        print("🟡 Partially unfroze shared layers")

    if episode == 1000:
        agent.freeze_shared(False)
        print("🔓 Fully unfroze shared layers for fine-tuning")

    # ---- Logging ----
    episode_returns.append(episode_reward)
    if terminated:
        success_count += 1
        success = True
    episode_success = 1 if success else 0
    success_log.append(episode_success)    
    hidden = agent.model.init_hidden(batch_size=1, device=DEVICE)

    if (episode + 1) % 10 == 0:
        torch.mps.empty_cache()  
        avg_return = np.mean(episode_returns[-10:])
        success_rate = success_count / (episode + 1)
        print(f"Episode {episode+1}/{NUM_EPISODES} | "
              f"SuccessRate={success_rate*100:.1f}% | ")
    else:
        print(f"Episode {episode+1}/{NUM_EPISODES} | Reward: {episode_reward:.2f}"
              f" | Success: {success}")

# --- SAVE TRAINED AGENT ---
new_checkpoint = os.path.join(checkpoint_dir, f"ppo_{env.spec.id.lower().replace('-','_')}.pt")
agent.save(new_checkpoint)

# --- Log metrics ---
df = pd.DataFrame({
    "Episode": np.arange(1, len(success_log) + 1),
    "Return": episode_returns,
    "Success": success_log
})
df["SuccessRate"] = np.cumsum(df["Success"]) / df["Episode"]

csv_path = os.path.join(checkpoint_dir, f"{env.spec.id}_transfer_metrics.csv")
df.to_csv(csv_path, index=False)
print(f"📊 Metrics saved to {csv_path}")

# ---- Plot training curve ----
plt.figure()
plt.plot(df["Episode"], df["SuccessRate"], label="Episode Success")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.title(f"PPO Transfer Learning - {env.spec.id}")
plt.legend()
plt.show()


env.close()
