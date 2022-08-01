import time

import gym

num_steps = 1500

env = gym.make('MountainCar-v0', new_step_api=True, render_mode="human")
obs = env.reset()

for step in range(num_steps):
    # Take random action
    action = env.action_space.sample()
    
    # Apply action
    # Env also rendered here?
    new_obs, reward, terminated, truncated, info = env.step(action)
    
    # Wait a bit
    time.sleep(0.001)
    
    # If episode done, start another one
    if terminated:
        env.reset()
        
env.close()