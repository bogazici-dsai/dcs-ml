from hirl.environments.HarfangEnv_GYM import HarfangEnv

env = HarfangEnv()
obs = env._get_observation()
print(obs)
