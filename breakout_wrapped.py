"""Using Breakout to help appreciate the Wrapper class' utility.

Running the environment with random actions.
"""

import time

import gym

from wrappers import ObservationWrapper, RewardWrapper, ActionWrapper

env = gym.make('ALE/Breakout-v5', new_step_api=True, render_mode="human")  # check render if slider moves left
wrapped_env = ObservationWrapper(RewardWrapper(ActionWrapper(env)))

obs = env.reset()

for step in range(500):
    action = wrapped_env.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped_env.step(action)

    # Raise flag if values not vectorised properly
    if (obs > 1).any() or (obs < 0).any():
         print("Max or min value of observation out of range")

    # Raise flag if reward not clipped
    if reward < 0 or reward > 1:
        assert False, "Reward out of bounrds"

    time.sleep(0.001)

wrapped_env.close()

print("All checks passed")
