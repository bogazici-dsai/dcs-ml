# make_minigrid_env.py
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
#from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import yaml
from gymnasium.wrappers import TimeLimit, FlattenObservation
from env.hirl.environments.HarfangEnv_GYM_new import HarfangEnv
from env.hirl.environments import dogfight_client as df
def make_env(max_steps: int = 5000, config_path: str = "env/local_config.yaml", port: int = None):
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

    """
        Creates and wraps the HARFANG environment.
        """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ip = str(cfg.get("network", {}).get("ip", "")).strip()
    port_cfg = int(cfg.get("network", {}).get("port", 50888))
    render = bool(cfg.get("render", True))

    # CLI port overrides config
    if port is not None:
        port_cfg = int(port)

    if not ip or ip == "YOUR_IP_ADDRESS":
        raise ValueError(f"Please set a valid 'network.ip' in {config_path}")

    # Connect once before env creation
    df.connect(ip, port_cfg)
    df.disable_log()
    df.set_renderless_mode(not render)  # correct passthrough
    df.set_client_update_mode(True)

    # Dict obs -> Flatten for MlpPolicy, with time limit + monitor
    env = HarfangEnv()
    env.reset()
    env = FlattenObservation(env)
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = Monitor(env)
    return env
