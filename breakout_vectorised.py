"""Render vectorised environments; display screenshots of the games in a tiled fashion."""

import time

import gym

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

def main(num_envs=3):
    envs = [lambda: gym.make('ALE/Breakout-v5', render_mode="human") for _ in range(num_envs)]

    # Vectorise envs
    envs = SubprocVecEnv(envs)

    # Get initial state
    init_obs = envs.reset()

    # Get list of obs corresponding to parallel envs
    print("Number of envs:", len(init_obs))

    # Inspect an obs
    ob_ind = 0
    print(f"Shape of env {ob_ind}:", init_obs[ob_ind].shape)

    for i in range(1000):
        actions = [envs.action_space.sample() for j in range(num_envs)]
        envs.step(actions)
        time.sleep(0.001)

    envs.close()

if __name__ == "__main__":
    main()
