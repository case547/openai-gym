"""Using Breakout to help appreciate the Wrapper class' utility.

Running the environment with random actions.
"""

import time

import gym

from wrappers import ConcatObs

env = gym.make('ALE/Breakout-v5', new_step_api=True, render_mode="human")

print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

obs = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.01)
env.close()