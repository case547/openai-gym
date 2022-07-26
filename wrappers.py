import numpy as np
import random
from collections import deque

import gym

class ConcatObs(gym.Wrapper):
    """Class to override reset() and return() methods of the Breakout env"""

    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env, new_step_api = True)
        self.k = k  # number of past frames to concatenate
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=((k,) + shp), dtype=env.observation_space.dtype
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            # No prev obs to concat, so just concat init obs repeatedly
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        return np.array(self.frames)


# Suppose have to make following changes:
#     Normalise pixel observations by 255
#     Clip rewards between 0 and 1
#     Prevent the slider from moving to the left (action 3).

class ObservationWrapper(gym.ObservationWrapper):
    """Helps make changes to the observation using the observation() method of the wrapper class."""
    
    def __init__(self, env):
        super().__init__(env, new_step_api = True)

    def observation(self, obs):
        # Normalise observation by 255
        return obs/255

class RewardWrapper(gym.RewardWrapper):
    """Helps make changes to the reward using the reward() method of the wrapper class."""

    def __init__(self, env):
        super().__init__(env, new_step_api = True)


    def reward(self, reward):
        # Clip reward to [0, 1]
        return np.clip(reward, 0, 1)

class ActionWrapper(gym.ActionWrapper):
    """Helps make changes to the action using the action() method of the wrapper class."""

    def __init__(self, env):
        super().__init__(env, new_step_api = True)

    def action(self, action):
        if action == 3:
            return random.choice([0,1,2])
        else:
            return action