import gym
import DCSGymEnv  # Make sure this file is named `DCSGymEnv.py`
import DCSRulePilot

env = DCSGymEnv.DCSGymEnv()
obs = env.reset()

done = False
while not done:
    action = DCSRulePilot.sample()  # Take a random action
    obs, reward, done, _ = env.step(action)
    print(f"Reward: {reward}, Obs: {obs}")

env.close()
