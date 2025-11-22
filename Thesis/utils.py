import numpy as np
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from ppo_lstm_agent import PPOAgent
from curiosity import CuriosityModule, NoCuriosity
import gymnasium as gym


def build_state(agent_pos, goal_pos, env, norm_scale=12.0):
    yaw = env.unwrapped.agent.dir
    rel_x = (goal_pos[0] - agent_pos[0]) / norm_scale
    rel_z = (goal_pos[2] - agent_pos[2]) / norm_scale
    
    # Add velocity information if available
    state = np.array([
        agent_pos[0] / norm_scale, 
        agent_pos[2] / norm_scale,
        goal_pos[0] / norm_scale, 
        goal_pos[2] / norm_scale,
        np.cos(yaw), 
        np.sin(yaw),
        rel_x, 
        rel_z
    ], dtype=np.float32)
    
    return state

def compute_reward(prev_pos, curr_pos, goal_pos, visited, norm_scale=12.0):
    """Reward function for early curriculum stages (1–2).
    Encourages distance improvement, exploration, and reaching the goal.
    Uses more shaping and smaller penalties for stable learning."""

    old_dist = np.linalg.norm(np.array(prev_pos) - np.array(goal_pos))
    new_dist = np.linalg.norm(np.array(curr_pos) - np.array(goal_pos))
    
    # Distance improvement (positive when getting closer)
    delta = (old_dist - new_dist) / norm_scale
    reward = 2.0 * delta

    # Small step penalty to encourage efficient movement
    reward -= 0.005

    # Exploration bonus for visiting new ground
    pos_key = (round(curr_pos[0], 1), round(curr_pos[2], 1))
    if pos_key not in visited:
        reward += 0.02
        visited.add(pos_key)

    # Success bonus for reaching the goal
    if new_dist < 0.4:
        reward += 15.0

    # Clip rewards to avoid instability
    return reward #float(np.clip(reward, -1.0, 1.0))

def compute_enhanced_reward(prev_pos, curr_pos, goal_pos, visited, steps_taken, max_steps=500):
    """Reward function for later curriculum stages (3–5).
    Reduces shaping, adds efficiency and final-approach scaling.
    Suitable once the agent can already navigate stably."""

    old_dist = np.linalg.norm(np.array(prev_pos) - np.array(goal_pos))
    new_dist = np.linalg.norm(np.array(curr_pos) - np.array(goal_pos))

    # 1. Base distance improvement
    distance_improvement = old_dist - new_dist
    distance_reward = 0.8 * distance_improvement  # slightly reduced shaping

    # 2. Stronger bonus for final approach
    final_approach_bonus = 0.0
    if new_dist < 1.0:
        # Exponential-style bonus that grows as agent gets closer
        closeness_bonus = (1.0 / (new_dist + 0.1)) * 0.05
        final_approach_bonus = closeness_bonus

    # 3. Success bonus for actually reaching the goal
    success_bonus = 0.0
    if new_dist < 0.4:
        efficiency = 1.0 - (steps_taken / max_steps)
        success_bonus = 15.0 + (5.0 * efficiency) 

    # 4. Mild exploration bonus (less important in later stages)
    pos_key = (round(curr_pos[0], 1), round(curr_pos[2], 1))
    exploration_bonus = 0.01 if pos_key not in visited else 0.0

    # 5. Time penalty that shrinks near the goal
    time_penalty = 0.01
    if new_dist < 2.0:
        time_penalty *= 0.1

    total_reward = (
        distance_reward
        + final_approach_bonus
        + success_bonus
        + exploration_bonus
        - time_penalty
    )

    return float(np.clip(total_reward, -1.0, 10.0))

def get_latest_checkpoint(path):
    files = [f for f in os.listdir(path) if f.endswith(".pt")]
    if not files:
        return None
    files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(path, f)), reverse=True)
    return os.path.join(path, files[0])

def train(env, num_episodes, agent, device, rollout):

    success_count = 0
    success_log = []

    #plt.ion()
    #fig, (ax_top, ax_fp) = plt.subplots(1, 2, figsize=(8, 4))

    USE_CURIOSITY = True  #toggle curiosity on and off

    if USE_CURIOSITY:
        curiosity_module = CuriosityModule(
            state_dim=8,
            num_actions=env.action_space.n,
            hidden_dim=128,
            device=device,
            learning_rate=1e-4,
            scale=0.01   # adjust if too strong
        )
    else:
        curiosity_module = NoCuriosity()

    for episode in range(num_episodes):
        hidden = agent.model.init_hidden(batch_size=1, device=device)
        obs = env.reset()
        start_of_episode = True

        agent_pos = env.unwrapped.agent.pos
        goal_pos = env.unwrapped.box.pos
        positions = []  # reset path tracker
        visited = set()

        state = build_state(agent_pos, goal_pos, env)

        done = False
        episode_reward = 0
        steps_in_episode = 0
        intrinsic_rewards = []
            
        while not done:
            # ---- Agent action ----
            action, log_prob, value, hidden = agent.act(state, hidden)
            prev_pos = env.unwrapped.agent.pos.copy()
            positions.append((env.unwrapped.agent.pos[0], env.unwrapped.agent.pos[2]))

            next_obs, _, terminated, truncated, _ = env.step(action)

            # Keep memory content, drop autograd history
            done = terminated or truncated
            curr_pos = env.unwrapped.agent.pos.copy()

            # ---- Reward shaping ----
            reward = compute_reward(prev_pos, curr_pos, goal_pos, visited) #for earlier stages
            #reward = compute_enhanced_reward(prev_pos, curr_pos, goal_pos, visited, steps_in_episode) #for later, harder stages
            next_state = build_state(curr_pos, goal_pos, env)

            intrinsic_reward = 0.0

            intrinsic_reward = curiosity_module.compute_intrinsic_reward(state, action, next_state) * 0.5
            intrinsic_rewards.append(intrinsic_reward)
            
            # ---- Combined reward ----
            total_reward = reward + intrinsic_reward

            agent.remember(state, action, log_prob, total_reward, done, value, hidden, start_of_episode)
            start_of_episode = False
            state = next_state
            episode_reward += total_reward
            steps_in_episode += 1

            # ---- PPO update when buffer full ----
            if len(agent.obs_buf) >= rollout:
                agent.update()

            #env.render()
            #top_view = env.unwrapped.render_top_view()   # top-down RGB array
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

        # ---- Logging ----
        episode_success = 1 if terminated else 0
        success_count += episode_success
        success_log.append(episode_success)    
        hidden = agent.model.init_hidden(batch_size=1, device=device)

        if (episode + 1) % 10 == 0:
            success_rate = success_count / (episode + 1)
            print(f"Episode {episode+1}/{num_episodes} | "
                f"SuccessRate={success_rate*100:.1f}% | ")
        else:
            print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f}"
                f" | Success: {bool(episode_success)}")
    return success_log

def log_metrics(success_log, env, checkpoint_dir):
    # --- Log metrics ---
    df = pd.DataFrame({
        "Episode": np.arange(1, len(success_log) + 1),
        "Success": success_log
    })
    df["SuccessRate"] = np.cumsum(df["Success"]) / df["Episode"]

    csv_path = os.path.join(checkpoint_dir, f"{env.spec.id}_transfer_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"📊 Metrics saved to {csv_path}")
    return df

def plot_training_curve(df, env):
    # ---- Plot training curve ----
    plt.figure()
    plt.plot(df["Episode"], df["SuccessRate"], label="Episode Success")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title(f"PPO Transfer Learning - {env.spec.id}")
    plt.legend()
    plt.show()



# --- custom render composite ---
#plt.ion()
#fig, (ax_top, ax_fp) = plt.subplots(1, 2, figsize=(8, 4))

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


def train_spat(env, num_episodes, agent, device, rollout, agent_pos, goal_pos):
    success_count = 0
    success_log = []
    steps_since_update = 0

    for episode in range(num_episodes):

        # ---- Reset environment ----
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        visited = set()
        step = 0

        # ---- Reset recurrent states (on device) ----
        memory, conv_state, lstm_state = agent.model.init_states(batch_size=1, device=device)
        prev_pos = env.unwrapped.agent.pos.copy()

        while not done and step < rollout:
            agent_pos = env.unwrapped.agent.pos
            goal_pos = env.unwrapped.box.pos

            # build state and pos coords (state on device, pos on device)
            state = build_spat_state(agent_pos, goal_pos, env).unsqueeze(0).to(device)
            pos_coords = build_position_coords(agent_pos, env).unsqueeze(0).to(device)

            # ---- capture hidden BEFORE forward (clone+detach to freeze)
            hidden_before = {
                "memory": memory.clone().detach(),
                "conv_h": conv_state[0].clone().detach(),
                "conv_c": conv_state[1].clone().detach(),
                "lstm_h": lstm_state[0].clone().detach(),
                "lstm_c": lstm_state[1].clone().detach(),
            }

            # ---- Step using spatial PPO agent (act uses device tensors) ----
            action, logp, value, memory, conv_state, lstm_state = agent.act(
                state,
                memory,
                conv_state,
                lstm_state,
                pos_coords
            )

            # ---- Step environment  ----
            next_obs, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            curr_pos = env.unwrapped.agent.pos
            agent.model.total_env_steps += 1

            # ---- Reward shaping (include env-provided reward) ----
            reward = compute_reward(prev_pos, curr_pos, goal_pos, visited)
            prev_pos = curr_pos.copy()

            # ---- Store transition for PPO using hidden_before ----
            agent.store(
                obs=state.squeeze(0),        
                position=pos_coords.squeeze(0),         
                action=action,
                logp=logp,                   
                reward=reward,
                done=done,
                value=value,
                hidden=hidden_before         
            )

            episode_reward += reward

            # ---- move to next state----
            step += 1
            steps_since_update += 1

            # ---- Trigger PPO update when we have rollout_len samples ----
            if len(agent.obs_buf) >= agent.rollout_len:
                agent.update()
                steps_since_update = 0

        # ---- Episode logging ----
        success = 1 if done and terminated else 0
        success_count += success
        success_log.append(success)

        if (episode+1) % 10 == 0:
            print(
                f"Episode {episode+1}/{num_episodes} | "
                f"Reward: {episode_reward:.2f} | Success: {bool(success)} | "
                f"SuccessRate={(success_count/(episode+1))*100:.1f}%"
            )

    return success_log

def build_spat_state(agent_pos, goal_pos, env, grid_size=16):
    min_x, max_x = env.unwrapped.min_x, env.unwrapped.max_x
    min_z, max_z = env.unwrapped.min_z, env.unwrapped.max_z

    state_map = np.zeros((3, grid_size, grid_size), dtype=np.float32)

    def to_grid(pos):
        gx = (pos[0] - min_x) / (max_x - min_x)
        gz = (pos[2] - min_z) / (max_z - min_z)
        gx = int(gx * (grid_size - 1))
        gz = int(gz * (grid_size - 1))
        return np.clip(gx, 0, grid_size - 1), np.clip(gz, 0, grid_size - 1)

    agent_x, agent_z = to_grid(agent_pos)
    goal_x, goal_z = to_grid(goal_pos)

    state_map[0, agent_z, agent_x] = 1.0
    state_map[1, goal_z, goal_x] = 1.0

    heading = env.unwrapped.agent.dir
    fov = np.deg2rad(90)
    n_rays = 19
    max_range = 4.0
    steps = 25

    for k in range(n_rays):
        angle = heading + fov * (k / (n_rays - 1) - 0.5)
        dx = np.cos(angle)
        dz = np.sin(angle)
        for t in range(1, steps + 1):
            dist = (t / steps) * max_range
            wx = agent_pos[0] + dx * dist
            wz = agent_pos[2] + dz * dist
            if wx < min_x or wx > max_x or wz < min_z or wz > max_z:
                break
            gx, gz = to_grid([wx, 0, wz])
            state_map[2, gz, gx] = 1.0
    state_map[2, agent_z, agent_x] = 1.0

    return torch.from_numpy(state_map).float()

def build_position_coords(agent_pos, env):
    min_x, max_x = env.unwrapped.min_x, env.unwrapped.max_x
    min_z, max_z = env.unwrapped.min_z, env.unwrapped.max_z
    x = (agent_pos[0] - min_x) / (max_x - min_x)
    z = (agent_pos[2] - min_z) / (max_z - min_z)
    x = np.clip(x, 0.01, 0.99)
    z = np.clip(z, 0.01, 0.99)
    return torch.tensor([x,z], dtype=torch.float32)