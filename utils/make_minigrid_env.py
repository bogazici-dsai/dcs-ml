# make_minigrid_env.py
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

def make_env(env_name="MiniGrid-DoorKey-6x6-v0", max_steps=100):
    # Create two separate env instances
    # TODO:
    # env = HarfangEnv()
    env_rl = gym.make(env_name, render_mode="rgb_array", max_steps=max_steps)
    env_llm = gym.make(env_name, render_mode="rgb_array", max_steps=max_steps)

    # RL env: partial obs + image
    rl_env = RGBImgPartialObsWrapper(env_rl)
    rl_env = ImgObsWrapper(rl_env)
    rl_env = Monitor(rl_env)  # Optional but useful for logging

    # LLM env: image only
    llm_env = env_llm
    # TODO:
    # return env
    return rl_env, llm_env