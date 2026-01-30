
import os
import pandas as pd
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym

def trainA2C_RGB(env, num_episodes, agent, device, rollout):

    success_count = 0
    success_log = []
    recent_success_streak = 0
    
    # Just for logging
    reward_log = []
    step_log = []

    for episode in range(num_episodes):
        ep_idx = episode + 1
        hidden = agent.model.init_hidden(batch_size=1, device=device)
        obs, _ = env.reset() # env.reset() returns obs, info
        goal_pos = env.unwrapped.box.pos
        agent_pos = env.unwrapped.agent.pos.copy()
        prev_pos = agent_pos.copy()
        visited = set()

        done = False
        episode_reward = 0
        step = 0
            
        while not done:
            # ---- Agent action ----
            action, log_prob, value, next_hidden = agent.act(obs, hidden)
            
            # Step
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            curr_pos = env.unwrapped.agent.pos.copy()
            reward = compute_reward(prev_pos, curr_pos, goal_pos, visited, terminated)
            
            # Store
            # Note: storing 'obs' (current), not next_obs.
            agent.store(obs, action, reward, done, value) 
            
            obs = next_obs
            hidden = next_hidden
            episode_reward += reward
            step += 1
            prev_pos = curr_pos

            # ---- Update ----
            if len(agent.obs_buf) >= rollout:
                agent.update(next_hidden, done=done)

        
        if len(agent.obs_buf) > 0:
             agent.update(hidden, done=True, force=True) # Update with whatever is left

        success = int(terminated)
        success_count += success
        success_log.append(success)    
        reward_log.append(episode_reward)
        step_log.append(step)

        success_rate = sum(success_log) / len(success_log)
        recent_success = sum(success_log[-100:]) / min(100, len(success_log))
        if recent_success > 0.8:
            recent_success_streak += 1
        else:
            recent_success_streak = 0

        if ep_idx % 100 == 0:
            print(f"Episode {ep_idx}/{num_episodes} | "
                  f"SuccessRate={success_rate*100:.1f}% | "
                  f"RecentSuccess={recent_success*100:.1f}% | "
                  f"AvgReward={np.mean(reward_log[-100:]):.3f} | "
                  f"AvgSteps={np.mean(step_log[-100:]):.1f}")
            if recent_success > 0.8:
                recent_success_streak += 1
            else:
                recent_success_streak = 0

        if recent_success_streak >= 5:
            print("Early stopping, recent success > 80% for 500 episodes in a row!")
            break
        
        # Save occasionally
        if ep_idx % 500 == 0:
             # save logic handled in main usually, but utils has agent access
             pass

    return success_log

def compute_reward(prev_pos, curr_pos, goal_pos, visited, terminated):
    reward = -0.01 # Time penalty

    # --- 1. Progress Shaping ---
    old_dist = np.linalg.norm(prev_pos - goal_pos)
    new_dist = np.linalg.norm(curr_pos - goal_pos)
    reward += 0.3 * (old_dist - new_dist)

    # --- 2. THE VISITED LOGIC ---
    #pos_key = (round(curr_pos[0] * 2) / 2, round(curr_pos[2] * 2) / 2)
    #if pos_key not in visited:
    #    reward += 0.05
    #    visited.add(pos_key)

    # --- 3. Goal Bonus ---
    if terminated:
        reward += 1.0

    return float(reward)

def get_latest_checkpoint(path):
    files = [f for f in os.listdir(path) if f.endswith(".pt")]
    if not files:
        return None
    files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(path, f)), reverse=True)
    return os.path.join(path, files[0])

def log_metrics(success_log, env, checkpoint_dir):
    df = pd.DataFrame({
        "Episode": np.arange(1, len(success_log) + 1),
        "Success": success_log
    })
    df["SuccessRate"] = df["Success"].expanding(min_periods=1).mean()

    csv_path = os.path.join(checkpoint_dir, f"{env.spec.id}_lstm_rgb_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"📊 Metrics saved to {csv_path}")
    return df

def plot_training_curve(df, env):
    plt.figure()
    plt.plot(df["Episode"], df["SuccessRate"], label="Success Rate (MA 100)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title(f"LSTM RGB - {env.spec.id}")
    plt.legend()
    plt.show()
