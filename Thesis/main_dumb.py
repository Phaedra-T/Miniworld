import gymnasium as gym
import miniworld
#from my_env import MySimpleEnv

if __name__ == '__main__' :
    env = gym.make("MiniWorld-Hallway-v0", render_mode = 'human') # max_episode_steps=
    obs, info = env.reset(seed=42)
    done = False

    while not done:
        action = env.action_space.sample()
        #action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render() #render_top_view()

        if reward > 0:
            print(f"🎉 Reached the goal! Reward: {reward:.2f}")

        done = terminated or truncated

    env.close()