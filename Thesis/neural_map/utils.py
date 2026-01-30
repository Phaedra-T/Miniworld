import os
import pandas as pd
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def train_nm(env, num_episodes, agent, device, rollout, goal_pos, save_dir, seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    success_log = []
    reward_log = []
    step_log = []
    rollout_steps = 0
    recent_success_streak = 0

    for episode in range(num_episodes):
        ep_idx = episode + 1
        obs, _ = env.reset()
        goal_pos_ep = env.unwrapped.box.pos if goal_pos is None else goal_pos
        done = False
        episode_reward = 0.0
        step = 0
        agent_pos = env.unwrapped.agent.pos
        prev_pos = agent_pos

        state = agent.model.init_states(batch_size=1, device=device)
        agent_pos = env.unwrapped.agent.pos
        prev_pos = agent_pos

        while not done:
            obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            pos_t = build_position_coords(agent_pos, env).to(device).unsqueeze(0)

            disable_write = agent.model.total_env_steps < agent.model.write_warmup_steps

            heading = float(env.unwrapped.agent.dir)
            heading_t = torch.tensor(heading, device=device, dtype=torch.float32)

            state_before = {"memory": state["memory"].detach().clone()}

            # --- act ---
            action, logp, value, state, _ = agent.act(
                obs_t, state, pos_t, heading=heading_t, disable_write=disable_write
            )

            # --- distance target ---
            dist = np.linalg.norm(env.unwrapped.agent.pos - goal_pos_ep)
            max_dist = np.sqrt(2) * (env.unwrapped.max_x - env.unwrapped.min_x)
            dist_norm = dist / max_dist


            # --- env step ---
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.model.total_env_steps += 1
            step += 1
            agent_pos = env.unwrapped.agent.pos
            reward = compute_reward(prev_pos, agent_pos, goal_pos_ep, terminated)
            episode_reward += reward

            # --- store ---
            agent.store(obs=obs_t.squeeze(0), position=pos_t.squeeze(0), heading=heading_t, action=action, reward=reward, done=done, value=value, state_before=state_before, dist=dist_norm)

            # --- build next state tensors ---
            prev_pos = agent_pos
            obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            pos_t = build_position_coords(agent_pos, env).to(device).unsqueeze(0)

            heading = float(env.unwrapped.agent.dir)
            heading_t = torch.tensor(heading, device=device, dtype=torch.float32)

            rollout_steps += 1

            if rollout_steps >= rollout or done:
                metrics = agent.update(next_obs=obs_t, next_state=state, next_pos=pos_t, next_head=heading_t, disable_write=disable_write)
                rollout_steps = 0
                if metrics:
                    if not hasattr(agent, "metrics_log"):
                        agent.metrics_log = []
                    agent.metrics_log.append(metrics)

        success = int(terminated)
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
            avg_reward = sum(reward_log[-100:]) / 100
            avg_steps = sum(step_log[-100:]) / 100

            dist_loss = agent.metrics_log[-1]["dist_loss"] if hasattr(agent, "metrics_log") and agent.metrics_log else 0.0
            mem_struct_loss = agent.metrics_log[-1]["mem_structure_loss"] if hasattr(agent, "metrics_log") and agent.metrics_log else 0.0
            grad_norm = agent.metrics_log[-1]["grad_norm"] if hasattr(agent, "metrics_log") and agent.metrics_log else 0.0

            print(
                f"Ep {ep_idx:04d} | "
                f"Overall Success:{success_rate*100:5.1f}% | "
                f"Recent Success:{recent_success*100:5.1f}% | "
                f"Avg Reward:{avg_reward:7.3f} | "
                f"Avg Steps:{avg_steps:3.1f} | "
                f"Dist_loss:{dist_loss:.4f} | "
                f"MemStruct:{mem_struct_loss:.4f} | "
                f"GradNorm:{grad_norm:.2f}"
            )

        if recent_success_streak >= 500:
            print("Early stopping, recent success > 80% for 500 episodes in a row!")
            break

    return success_log

def compute_reward(prev_pos, curr_pos, goal_pos, terminated):
    reward = -0.01 
    
    old_dist = np.linalg.norm(prev_pos - goal_pos)
    new_dist = np.linalg.norm(curr_pos - goal_pos)
    reward += 0.3 * (old_dist - new_dist)
    if terminated:
        reward += 1.0

    return float(reward)


def build_position_coords(agent_pos, env):
    min_x = env.unwrapped.min_x
    max_x = env.unwrapped.max_x
    min_z = env.unwrapped.min_z
    max_z = env.unwrapped.max_z

    width = max_x - min_x
    depth = max_z - min_z

    x = (agent_pos[0] - min_x) / (width + 1e-5)
    z = (agent_pos[2] - min_z) / (depth + 1e-5)

    if x > 1 or x < 0 or z > 1 or z < 0:
        print(f"Write center (normed): x={x:.3f}, y={z:.3f}")
    return torch.tensor([x, z], dtype=torch.float32)


def get_latest_checkpoint(path):
    files = [f for f in os.listdir(path) if f.endswith(".pt")]
    if not files:
        return None
    files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(path, f)), reverse=True)
    return os.path.join(path, files[0])


def log_metrics(success_log, env, checkpoint_dir):
    df = pd.DataFrame({
        "Episode": np.arange(1, len(success_log) + 1),
        "Success": success_log,
    })
    df["SuccessRate"] = np.cumsum(df["Success"]) / df["Episode"]

    csv_path = os.path.join(checkpoint_dir, f"{env.spec.id}_transfer_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"📊 Metrics saved to {csv_path}")
    return df


def plot_training_curve(df, env):
    plt.figure()
    plt.plot(df["Episode"], df["SuccessRate"], label="Episode Success")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title(f"Transfer Learning - {env.spec.id}")
    plt.legend()
    plt.show()


def save_memory_snapshot(memory_tensor, episode_idx, save_dir):
    if memory_tensor.dim() == 4:
        mem = memory_tensor.squeeze(0)
    else:
        mem = memory_tensor

    mem_norm = mem.norm(dim=0).detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(mem_norm, cmap="inferno", origin="lower")
    ax.set_title("Memory Map (Norm L2)")
    plt.colorbar(im, ax=ax)

    plt.suptitle(f"Neural Map Ep {episode_idx}")
    filename = os.path.join(save_dir, f"memory_ep_{episode_idx:04d}.png")
    plt.savefig(filename)
    plt.close()
