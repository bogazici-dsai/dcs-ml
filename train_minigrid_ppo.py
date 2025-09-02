import os
import glob
import argparse
import time

from click import progressbar

import wandb
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback

import yaml
from gymnasium.wrappers import TimeLimit, FlattenObservation
from env.hirl.environments.HarfangEnv_GYM_new import HarfangEnv
from env.hirl.environments import dogfight_client as df

'''# ------------------------------ ENV FACTORY ------------------------------ #'''
def make_env(max_steps: int = 5000, config_path: str = "env/local_config.yaml", port: int = None):
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
    df.set_renderless_mode(not render)
    df.set_client_update_mode(True)

    # Dict obs -> Flatten for MlpPolicy, with time limit
    env = HarfangEnv()
    env.reset()
    env = FlattenObservation(env)
    env = TimeLimit(env, max_episode_steps=max_steps)
    return env


'''# ------------------------------ UTILITIES ------------------------------ #'''
def select_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.backends.cuda.is_built() and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
def list_models(models_dir: str):
    """
    Returns list of .zip models sorted by mtime (newest first).
    """
    paths = glob.glob(os.path.join(models_dir, "*.zip"))
    paths = [(p, os.path.getmtime(p)) for p in paths]
    paths.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in paths]
def resolve_model_path(args) -> str:
    """
    Model selection logic for prediction:
    1) If --model-path is provided and exists -> use it.
    2) Else pick the Nth most recent model in --models-dir (default pick=0, i.e., latest).
    3) If none exist -> raise a clear error.
    """
    if args.model_path:
        if os.path.isfile(args.model_path):
            print(f"Loading model from explicit path: {args.model_path}")
            return args.model_path
        raise FileNotFoundError(f"--model-path not found: {args.model_path}")

    candidates = list_models(args.models_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No model files found in '{args.models_dir}'. "
            f"Train first or pass --model-path /path/to/model.zip"
        )

    pick = max(0, int(args.pick))
    if pick >= len(candidates):
        print(f"--pick={pick} is out of range (only {len(candidates)} models). Using latest instead.")
        pick = 0

    # Show a friendly short list
    print("\nAvailable models (newest first):")
    for i, p in enumerate(candidates[:10]):  # show top 10
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
        marker = "<-- pick" if i == pick else ""
        print(f"  [{i}] {os.path.basename(p)}  |  {t}  {marker}")
    chosen = candidates[pick]
    print(f"\nSelected model: {chosen}\n")
    return chosen


'''# ------------------------------ ARG PARSER ------------------------------ #'''
def build_parser():
    parser = argparse.ArgumentParser(description="Harfang PPO (train/predict) with W&B logging")

    # Mode
    parser.add_argument("--mode", choices=["train", "predict"], default="train",
                        help="Run mode: train or predict")

    # Env / rollout controls
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Max steps per episode (TimeLimit).")
    parser.add_argument("--total-steps", type=int, default=2_000_000,
                        help="Total training timesteps for PPO when --mode=train.")
    parser.add_argument("--config-path", type=str, default="env/local_config.yaml",
                        help="YAML config for environment connection/settings.")
    parser.add_argument("--port", type=int, default=None,
                        help="Override port from config file.")

    # Models
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Directory to save/load models.")
    parser.add_argument("--save-name", type=str, default="ppo_harfang_v3.zip",
                        help="Final model filename for training.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="(Predict) explicit path to a .zip model (overrides --pick).")
    parser.add_argument("--pick", type=int, default=0,
                        help="(Predict) pick Nth most recent model (0=latest).")

    # Prediction settings
    parser.add_argument("--episodes", type=int, default=5,
                        help="(Predict) number of episodes to run.")
    parser.add_argument("--deterministic", default=True, action="store_true",
                        help="(Predict) use deterministic actions (default is stochastic).")

    # WandB
    parser.add_argument("--project", type=str, default="Harfang_PURE_RL")
    parser.add_argument("--entity", type=str, default="BILGEM_DCS_RL")
    parser.add_argument("--run-name", type=str, default=None)

    return parser

'''# ------------------------------ MAIN ------------------------------ #'''
def main():
    args = build_parser().parse_args()

    # Device
    device = select_device()
    print("Using device:", device)

    # Env
    env = make_env(max_steps=args.max_steps, config_path=args.config_path, port=args.port)

    # Ensure model dir exists
    os.makedirs(args.models_dir, exist_ok=True)

    # ---------------------------- TRAIN ---------------------------- #
    if args.mode == "train":
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=args.run_name or f"harfang_train_{time.strftime('%Y%m%d_%H%M%S')}",
            config={
                "mode": "train",
                "env_name": "Harfang",
                "algo": "PPO",
                "device": str(device),
                "total_timesteps": int(args.total_steps),
                "max_steps": int(args.max_steps),
            },
            sync_tensorboard=True,
        )

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_logs/",
            device=device
        )

        # Callbacks
        wandb_callback = WandbCallback(
            gradient_save_freq=1000,
            model_save_path=None,
            verbose=2
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,
            save_path=args.models_dir,   # consistent relative path
            name_prefix="ppo_harfang_v3_checkpoint"
        )

        # Train (SB3 logs to TensorBoard; W&B syncs TB scalars automatically)
        model.learn(
            total_timesteps=int(args.total_steps),
            callback=[checkpoint_callback, wandb_callback],
            progress_bar=True
        )

        # Save final model
        final_path = os.path.join(args.models_dir, args.save_name)
        model.save(final_path)
        print(f"\n✅ PPO model saved to: {final_path}")
        wandb.summary["final_model_path"] = final_path
        run.finish()

    # --------------------------- PREDICT --------------------------- #
    elif args.mode == "predict":
        model_path = resolve_model_path(args)
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=args.run_name or f"harfang_predict_{time.strftime('%Y%m%d_%H%M%S')}",
            config={
                "mode": "predict",
                "env_name": "Harfang",
                "algo": "PPO",
                "device": str(device),
                "episodes": int(args.episodes),
                "deterministic": bool(args.deterministic),
                "model_path": model_path,
                "max_steps": int(args.max_steps),
            },
        )

        model = PPO.load(model_path, device=device)
        print(f"✅ Model loaded from {model_path}\n")
        print("Starting prediction...")

        global_step = 0
        start_wall = time.time()

        for ep in range(1, int(args.episodes) + 1):
            obs, info = env.reset()
            done, truncated = False, False
            ep_reward = 0.0
            steps = 0
            ep_start = time.time()

            while not (done or truncated):
                # Default SB3 behavior: deterministic=False unless flag is passed
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, done, truncated, info = env.step(action)

                # per-step logging
                ep_reward += float(reward)
                steps += 1
                global_step += 1
                wandb.log({
                    "predict/step_reward": float(reward),
                    "predict/global_step": global_step,
                })

            ep_time = time.time() - ep_start
            fps = steps / ep_time if ep_time > 0 else 0.0

            # per-episode logging
            wandb.log({
                "predict/episode": ep,
                "predict/episode_return": ep_reward,
                "predict/episode_length": steps,
                "predict/episode_success": info.get("success"),
                "predict/episode_fps": fps,
                "predict/mean_reward_per_step": (ep_reward / max(1, steps)),
            })
            print(f"Episode {ep}/{args.episodes} -> steps={steps}  reward={ep_reward:.3f}  fps={fps:.1f}")

        total_wall = time.time() - start_wall
        wandb.summary["predict/total_wall_time_sec"] = total_wall
        run.finish()
        print("✅ Prediction finished.")


if __name__ == "__main__":
    main()
