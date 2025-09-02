# make_minigrid_env.py
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
#from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import yaml
from gymnasium.wrappers import TimeLimit, FlattenObservation
from env.hirl.environments.HarfangEnv_GYM_new import HarfangEnv
from env.hirl.environments import dogfight_client as df
def make_env(max_steps: int = 5000, config_path: str = "env/local_config.yaml"):
    # # Create two separate env instances

    # env_rl = gym.make(env_name, render_mode="rgb_array", max_steps=max_steps)
    # env_llm = gym.make(env_name, render_mode="rgb_array", max_steps=max_steps)
    #
    # # RL env: partial obs + image
    # rl_env = RGBImgPartialObsWrapper(env_rl)
    # rl_env = ImgObsWrapper(rl_env)
    # rl_env = Monitor(rl_env)  # Optional but useful for logging
    #
    # # LLM env: image only
    # llm_env = env_llm
    # return rl_env,

    # --- Read config ---

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ip = str(cfg.get("network", {}).get("ip", "")).strip()
    # port = int(cfg.get("network", {}).get("port", 50888))
    port = 50777
    render = bool(cfg.get("render", True))

    if not ip or ip == "YOUR_IP_ADDRESS":
        raise ValueError(f"Please set a valid 'network.ip' in {config_path}")

    # --- Connect df once before env creation ---
    df.connect(ip, port)
    df.disable_log()
    df.set_renderless_mode(not render)
    df.set_client_update_mode(True)

    # --- Build env chain (MlpPolicy needs flat Box obs) ---
    env = HarfangEnv()  # Dict obs
    env.reset()  # ensures observation_space is built
    env = FlattenObservation(env)
    env = TimeLimit(env, max_episode_steps=max_steps)  # truncated=True at horizon

    return env
