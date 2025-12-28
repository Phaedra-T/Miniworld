import os
import pandas as pd
import random
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
#from curiosity import CuriosityModule, NoCuriosity
import gymnasium as gym
from collections import deque, defaultdict


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

def compute_reward2(prev_pos, curr_pos, goal_pos, visited):
    """Reward scales with relative progress, not absolute distance."""

    old_dist = np.linalg.norm(np.array(prev_pos) - np.array(goal_pos))
    new_dist = np.linalg.norm(np.array(curr_pos) - np.array(goal_pos))

    scale = max(1.0, old_dist)   

    progress = (old_dist - new_dist) / scale
    reward = 0.2 * progress       # Well-scaled, not noisy

    reward -= 0.005              # small step penalty

    #pos_key = (round(curr_pos[0], 1), round(curr_pos[2], 1))
    #if pos_key not in visited:
    #    reward += 0.03
    #    visited.add(pos_key)

    if new_dist > 1.5:
        reward += 0.01 

    if new_dist < 0.4:
        reward += 15.0 


    return float(np.clip(reward, -2.0, 2.0))

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

    return float(np.clip(total_reward, -1.0, 2.0))

def get_latest_checkpoint(path):
    files = [f for f in os.listdir(path) if f.endswith(".pt")]
    if not files:
        return None
    files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(path, f)), reverse=True)
    return os.path.join(path, files[0])

#def train(env, num_episodes, agent, device, rollout):

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


def train_spat(env, num_episodes, agent, device, rollout, goal_pos):

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)

    success_count = 0
    success_log = []
    best_success = 0
    rollout_step = 0
    success_rate = 0
    episode = 0

    new_checkpoint = os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"),
    f"ppo_{env.spec.id.lower().replace('-', '_')} best.pt"
    )

    # --- custom render composite ---
    #plt.ion()
    #fig, (ax_top, ax_fp) = plt.subplots(1, 2, figsize=(8, 4))

    for episode in range(num_episodes):

        # ---- Reset environment ----
        obs, _ = env.reset()
        agent_pos = env.unwrapped.agent.pos.copy()
        prev_pos = env.unwrapped.agent.pos.copy()
        episode += 1

        episode_reward = 0.0
        done = False
        step = 0

        state = agent.model.init_states(batch_size=1, device=device)
        memory = state["memory"]

        while not done:

            # build state and pos coords (state on device, pos on device)
            state_tensor = build_spat_state(agent_pos, goal_pos, env).unsqueeze(0).to(device)
            pos_coords = build_position_coords(agent_pos, env).unsqueeze(0).to(device)
            disable_write = agent.model.total_env_steps < agent.model.write_warmup_steps

            memory_before = memory.clone().detach()
            # ---- ACT ----
            action, logp, value, state, logits, read = agent.act(
                state_tensor, state, pos_coords, disable_write=disable_write
                )

            # --------- STEP ENV ---------
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent_pos = env.unwrapped.agent.pos
            agent.model.total_env_steps += 1

            reward = compute_reward(prev_pos, agent_pos, goal_pos, terminated)
            prev_pos = agent_pos.copy()
            episode_reward += reward

            hidden = {
                "state": {k: v.detach().cpu() for k, v in state.items()},
                "episode_start": (step == 0),
                "disable_write": disable_write
            }
            
            # ---- Store transition for PPO using hidden_before ----
            agent.store(
                obs=state_tensor.squeeze(0),        
                position=pos_coords.squeeze(0),         
                action=action,
                logp=logp,      
                logits=logits,             
                reward=reward,
                done=done,
                value=value,
                hidden=hidden,
                mem_read=read,
                disable_write=disable_write
            )

            step += 1
            rollout_step += 1

            # --------- PPO UPDATE ---------
            if rollout_step % rollout == 0:
                agent.update()
                rollout_step = 0

            if done:
                break

            #env.render()
            # --- Custom render composite ---
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
 
        # ---- Episode logging ----
        success = int(terminated)
        success_count += success
        success_log.append(success)
        success_rate = (success_count/(episode+1))


        if (episode+1) % 10 == 0:
            print(
                f"Episode {episode+1}/{num_episodes} | "
                f"Reward: {episode_reward:.2f} | Success: {bool(success)} | "
                f"SuccessRate={success_rate*100:.1f}%"
            )
            if (success_rate > best_success):
                    best_success = success_rate
                    agent.save(new_checkpoint)
                    print( "New best success! Saving checkpoint")
            if (success_rate) >= 0.8:
                print("Early stopping, reached goal succes rate!")
                break

            if episode % 100 == 0:
                mem_mean = memory.mean().item()
                mem_std = memory.std().item()
                print(f"Memory stats: mean={mem_mean:.3f}, std={mem_std:.3f}")
                        
        #if (episode+1) % 100 == 0:
            #print(f"Current LR: {agent.optimizer.param_groups[0]['lr']}")
           # mem_cpu = memory.squeeze(0).detach().cpu().numpy()   # [C,H,W]
            # sum across channels and plt.imshow
           # import matplotlib.pyplot as plt
           # plt.imshow(mem_cpu.sum(0))
            #plt.title(f"Episode {episode+1} memory sum")
            #plt.savefig(f"memory_{episode+1:04d}.png")
            #plt.close()

        #agent.scheduler.step(success_rate)

    return success_log

def build_spat_state(agent_pos, goal_pos, env, view_radius = 2, grid_size=32):

        size = 2 * view_radius + 1
        state = np.zeros((3, size, size), dtype=np.float32)

        min_x, max_x = env.unwrapped.min_x, env.unwrapped.max_x
        min_z, max_z = env.unwrapped.min_z, env.unwrapped.max_z

        # --- Heading ---
        dir_idx = int(env.unwrapped.agent.dir)
        n_dirs = 8
        heading = (dir_idx / n_dirs) * 2 * np.pi

        cos_h, sin_h = np.cos(heading), np.sin(heading)

        # --- Helper: world → egocentric grid ---
        def world_to_ego(wx, wz):
            dx = wx - agent_pos[0]
            dz = wz - agent_pos[2]

            # rotate into egocentric frame
            ex =  cos_h * dx + sin_h * dz
            ez = -sin_h * dx + cos_h * dz

            gx = int(round(ex)) + view_radius
            gz = int(round(ez)) + view_radius

            return gx, gz

        # --- Channel 0: local occupancy (free space proxy) ---
        # Empty room → constant 1.0 inside view
        state[0, :, :] = 1.0

        # --- Channel 1: goal visible only if inside view ---
        gx, gz = world_to_ego(goal_pos[0], goal_pos[2])
        if 0 <= gx < size and 0 <= gz < size:
            state[1, gz, gx] = 1.0

        # --- Channel 2: heading encoding (broadcast) ---
        state[2, :, :] = sin_h  # or cos_h if you prefer

        return torch.from_numpy(state).float()


def build_position_coords(agent_pos, env):
    min_x, max_x = env.unwrapped.min_x, env.unwrapped.max_x
    min_z, max_z = env.unwrapped.min_z, env.unwrapped.max_z
    x = (agent_pos[0] - min_x) / (max_x - min_x)
    z = (agent_pos[2] - min_z) / (max_z - min_z)
    x = np.clip(x, 0.0, 1.0)
    z = np.clip(z, 0.0, 1.0)
    return torch.tensor([x,z], dtype=torch.float32)

def compute_reward(prev_pos, curr_pos, goal_pos, terminated):
    prev_dist = np.linalg.norm(goal_pos - prev_pos)
    curr_dist = np.linalg.norm(goal_pos - curr_pos)

    reward = -0.01  # time penalty

    # progress shaping
    #progress = prev_dist - curr_dist
    #reward += 0.5 * np.clip(progress, -0.05, 0.05)

    if terminated:
        reward += 1.0  

    return float(reward)