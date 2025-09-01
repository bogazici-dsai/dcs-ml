import os
import wandb
# import minigrid
import gymnasium as gym

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
# from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

import yaml
from gymnasium.wrappers import TimeLimit, FlattenObservation
from env.hirl.environments.HarfangEnv_GYM_new import HarfangEnv
from env.hirl.environments import dogfight_client as df

# Connect, Create and Wrap the HARFANG environment
def make_env(max_steps: int = 5000, config_path: str = "env/local_config.yaml"):
    # --- Read config ---
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ip = str(cfg.get("network", {}).get("ip", "")).strip()
    port = int(cfg.get("network", {}).get("port", 50888))
    render = bool(cfg.get("render", True))

    if not ip or ip == "YOUR_IP_ADDRESS":
        raise ValueError(f"Please set a valid 'network.ip' in {config_path}")

    # --- Connect df once before env creation ---
    df.connect(ip, port)
    df.disable_log()
    df.set_renderless_mode(not render)
    df.set_client_update_mode(True)

    # --- Build env chain (MlpPolicy needs flat Box obs) ---
    env = HarfangEnv()      # Dict obs
    env.reset()             # ensures observation_space is built
    env = FlattenObservation(env)
    env = TimeLimit(env, max_episode_steps=max_steps)  # truncated=True at horizon

    return env

# PURE RL

if __name__ == "__main__":
    wandb.init(
        project="Harfang_PURE_RL",
        entity="BILGEM_DCS_RL",
        name=f"harfang_pure_rl_01-09-2025",
        config={
            "env_name": "Harfang",
            "total_timesteps": 50_000,
            "algo": "PPO",
            "max_steps": 5000
        },
        sync_tensorboard=True
    )

    env = make_env()

    # Define model save path
    os.makedirs("models", exist_ok=True)
    save_path = "models/ppo_harfang_v1.zip"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.backends.cuda.is_built() and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Define PPO model
    # TODO: CnnPolicy? or MlpPolicy (cnnpolicy is for image observations so mlppolicy is better for flatten observations)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_logs/",
        device=device
    )
    # Define Callbacks
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=None,
        verbose=2
    )

    # Checkpoint every N steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="/models/",  # required folder path
        name_prefix="ppo_harfang_v1_checkpoint"
    )

    # Train the model (
    total_timesteps = 5000000
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, wandb_callback])

    # Save the final model
    model.save(save_path)
    print(f"\n✅ PPO model saved to: {save_path}")
