# harfang_train_predict.py
# - TensorBoard sync: ON
# - No step= in wandb.log (avoids warning)
# - Custom RL-step axes defined via run.define_metric
# - Full obs rows appended EVERY env step; table flushed every TABLE_LOG_EVERY steps
# - Episode-end one-row snapshot
# - All W&B + TB files pinned to ephemeral dir (NOT OS temp) and deleted at run end
# - Plus: WANDB_DIR, WANDB_CACHE_DIR, WANDB_DATA_DIR, TMP, TEMP all set to ephemeral paths
import gymnasium as gym
import os
import glob
import argparse
import time
import shutil
import atexit
import tempfile
import numpy as np
import yaml

import wandb
import torch
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from wandb.integration.sb3 import WandbCallback

from gymnasium.wrappers import TimeLimit, FlattenObservation
from env.hirl.environments.HarfangEnv_GYM_new import HarfangEnv
from env.hirl.environments import dogfight_client as df
from stable_baselines3.common.monitor import Monitor
from Assistants.rl_llm_assistant import MinigridAgent
from Assistants.rl_llm_assistant import HarfangAgent

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda

# ------------------------------ CONFIG ------------------------------ #
TABLE_LOG_EVERY = 200  # flush obs table to W&B every N env steps (1 = every step). Raised to reduce churn.

# ------------------------------ EPHEMERAL W&B DIR ------------------------------ #
def _make_ephemeral_wandb_dir():
    base = os.path.abspath("wandb_ephemeral")
    os.makedirs(base, exist_ok=True)
    run_dir = tempfile.mkdtemp(prefix="run_", dir=base)
    print("[W&B] Ephemeral run dir:", run_dir)
    # subfolders we may need
    for sub in ("tmp", "cache", "data", "tb"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir

def _cleanup_dir(path, retries=8, delay=0.5):
    if not path or not os.path.exists(path):
        return
    for _ in range(retries):
        try:
            shutil.rmtree(path, ignore_errors=True)
            if not os.path.exists(path):
                print("[W&B] Cleaned:", path)
                return
        except Exception:
            pass
        time.sleep(delay)
    if os.path.exists(path):
        print("[W&B] WARN: could not remove:", path)

_WANDB_EPHEMERAL_DIR = _make_ephemeral_wandb_dir()

# Pin Python temp and W&B dirs to the ephemeral folder (prevents OS temp cleanups on Windows)
tempfile.tempdir = os.path.join(_WANDB_EPHEMERAL_DIR, "tmp")
os.makedirs(tempfile.tempdir, exist_ok=True)

os.environ["WANDB_DIR"]       = _WANDB_EPHEMERAL_DIR
os.environ["WANDB_CACHE_DIR"] = os.path.join(_WANDB_EPHEMERAL_DIR, "cache")
os.environ["WANDB_DATA_DIR"]  = os.path.join(_WANDB_EPHEMERAL_DIR, "data")
os.environ["TMP"]  = os.path.join(_WANDB_EPHEMERAL_DIR, "tmp")   # Windows temp
os.environ["TEMP"] = os.path.join(_WANDB_EPHEMERAL_DIR, "tmp")

atexit.register(lambda: _cleanup_dir(_WANDB_EPHEMERAL_DIR))

# ------------------------------ SAFE LOG WRAPPER ------------------------------ #
def safe_wandb_log(payload):
    try:
        wandb.log(payload)
    except Exception as e:
        print(f"[W&B WARN] log failed: {e}")

# ------------------------------ FULL OBS CALLBACK ------------------------------ #
class WandbFullObsLogger(BaseCallback):
    """
    TB sync ON version:
      - Appends a row EVERY ENV STEP: [timestep, episode, *obs...]
      - Flushes the growing table to W&B every `table_log_every` env steps
      - Logs a one-row episode-end snapshot (last obs of that episode)
      - Does NOT pass `step=` to wandb.log; includes 'train/rl_step' field
    """
    def __init__(self, table_log_every: int = TABLE_LOG_EVERY):
        super().__init__()
        self.table = None
        self.keys = None
        self.episode = 0
        self._last_row = None
        self._table_log_every = int(table_log_every)

    def _safe_float(self, v):
        try:
            return float(v)
        except Exception:
            return float("nan")

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", None)

        # Pull dict from base env
        try:
            sd_list = self.training_env.get_attr("state_dict")
            if not sd_list:
                return True
            sd = sd_list[0]  # single env
        except Exception:
            return True

        # Lazy init
        if self.table is None:
            self.keys = sorted(list(sd.keys()))
            self.table = wandb.Table(
                columns=["timestep", "episode"] + self.keys,
                log_mode="INCREMENTAL"
            )

        # Build row for THIS step
        cur_step = int(self.num_timesteps)
        row = [cur_step, int(self.episode)] + [self._safe_float(sd.get(k)) for k in self.keys]
        self.table.add_data(*row)
        self._last_row = row

        # Periodic flush of the growing table
        if (cur_step % self._table_log_every) == 0:
            safe_wandb_log({
                "train/rl_step": cur_step,
                "train/obs_full_table": self.table,
                "train/last_timestep": cur_step
            })

        # If any env finished -> episode-end snapshot
        if dones is not None and np.any(dones):
            ep_table = wandb.Table(columns=["timestep", "episode"] + self.keys, log_mode="MUTABLE")
            ep_table.add_data(*self._last_row)
            safe_wandb_log({
                "train/rl_step": cur_step,
                "train/obs_last_step_ep": ep_table
            })
            self.episode += int(np.sum(dones))
        return True

    def _on_training_end(self) -> None:
        if self.table is not None:
            cur_step = int(self.num_timesteps)
            safe_wandb_log({
                "train/rl_step": cur_step,
                "train/obs_full_table_final": self.table,
                "train/last_timestep": cur_step
            })

# ------------------------------ ENV FACTORY ------------------------------ #
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
    df.set_renderless_mode(not render)  # correct passthrough
    df.set_client_update_mode(True)

    # Dict obs -> Flatten for MlpPolicy, with time limit + monitor
    env = HarfangEnv()
    env.reset()
    env = FlattenObservation(env)
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = Monitor(env)
    return env

'''# ------------------------------ LLM ------------------------------ #'''
#llm_model_name = "MOCK"
# llm = "MOCK"


llm_model_name = "llama3.1:8b"
chat = ChatOllama(model=llm_model_name, temperature=0.0)
llm = RunnableLambda(lambda x: chat.invoke(x))

harfang_agent = HarfangAgent(llm=llm, verbose=True)

'''# ------------------------------ OVERRIDE-PATCH ------------------------------ #'''
# --- LLM override every N steps, buffer-safe ---

def patch_policy_forward_with_llm(model, override_agent, override_every=4, env_name="Harfang"):
    """
    PPO policy.forward is patched.
    Every N steps (override_every), PPO's action is overridden by LLM's suggestion.
    """
    orig_forward = model.policy.forward
    n_envs = getattr(model, "n_envs", 1)
    env_steps = np.zeros(n_envs, dtype=np.int64)

    def patched_forward(obs: th.Tensor, deterministic: bool = False):
        acts, vals, logp = orig_forward(obs, deterministic)

        if override_agent is None or override_every <= 0:
            env_steps[:] += 1
            return acts, vals, logp

        obs_np  = obs.detach().cpu().numpy()
        acts_np = acts.detach().cpu().numpy()

        if acts_np.ndim == 0:  # single-env case
            obs_np  = np.expand_dims(obs_np, 0)
            acts_np = np.array([acts_np])

        new_acts = acts_np.copy()

        for i in range(len(acts_np)):
            if env_steps[i] % override_every == 0:
                # ✅ Expect environment dict observation
                if hasattr(model.env, "envs"):
                    base_env = model.env.envs[i].unwrapped
                else:
                    base_env = model.env.unwrapped

                obs_dict = getattr(base_env, "state_dict", None)

                if obs_dict is None:
                    print("[LLM Override] WARNING: No dict observation available, skipping override.")
                    continue

                llm_act = override_agent.decide(obs_dict)

                # keep inside action space
                if hasattr(model.action_space, "n"):
                    nA = model.action_space.n
                    if llm_act < 0 or llm_act >= nA:
                        llm_act = llm_act % nA

                new_acts[i] = llm_act
                print(f"[LLM Override] env={i}, step={env_steps[i]}, "
                      f"policy_act={int(acts_np[i])}, llm_act={llm_act}")

        acts_over = th.as_tensor(new_acts, device=acts.device, dtype=acts.dtype)

        with th.no_grad():
            feats = model.policy.extract_features(obs)
            latent_pi, latent_vf = model.policy.mlp_extractor(feats)
            dist = model.policy._get_action_dist_from_latent(latent_pi)
            logp_over = dist.log_prob(acts_over)
            vals_over = model.policy.value_net(latent_vf)

        env_steps[:] += 1
        return acts_over, vals_over, logp_over

    model.policy.forward = patched_forward


# ------------------------------ UTILITIES ------------------------------ #
def select_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.backends.cuda.is_built() and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def list_models(models_dir: str):
    paths = glob.glob(os.path.join(models_dir, "*.zip"))
    paths = [(p, os.path.getmtime(p)) for p in paths]
    paths.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in paths]

def resolve_model_path(args) -> str:
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

    print("\nAvailable models (newest first):")
    for i, p in enumerate(candidates[:10]):
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
        marker = "<-- pick" if i == pick else ""
        print(f"  [{i}] {os.path.basename(p)}  |  {t}  {marker}")
    chosen = candidates[pick]
    print(f"\nSelected model: {chosen}\n")
    return chosen

# ------------------------------ ARG PARSER ------------------------------ #
def build_parser():
    parser = argparse.ArgumentParser(description="Harfang PPO (train/predict) with W&B + TB sync + robust Tables")
    # Mode
    parser.add_argument("--mode", choices=["train", "predict"], default="train", help="Run mode")
    # LLM Assist
    parser.add_argument(
        "--LLM_assist",
        action="store_true",
        help="Enable LLM-assisted override of PPO actions (every N steps)."
    )
    parser.add_argument(
        "--LLM_assist2",
        action="store_true",
        help="Enable LLM-assisted override of PPO actions (every N steps)."
    )
    # Env / rollout
    parser.add_argument("--max-steps", type=int, default=5000, help="Max steps per episode (TimeLimit).")
    parser.add_argument("--total-steps", type=int, default=2_000_000, help="Total PPO timesteps for training.")
    parser.add_argument("--config-path", type=str, default="env/local_config.yaml", help="Env YAML config path.")
    parser.add_argument("--port", type=int, default=None, help="Override port from config file.")
    # Models
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save/load models.")
    parser.add_argument("--save-name", type=str, default="ppo_harfang_v3.zip", help="Final model filename.")
    parser.add_argument("--model-path", type=str, default=None, help="(Predict) explicit model path.")
    parser.add_argument("--pick", type=int, default=0, help="(Predict) pick Nth most recent model (0=latest).")
    # Prediction
    parser.add_argument("--episodes", type=int, default=5, help="(Predict) number of episodes to run.")
    parser.add_argument("--deterministic", action="store_true", help="(Predict) use deterministic actions.")
    # W&B
    parser.add_argument("--project", type=str, default="Harfang_PURE_RL")
    parser.add_argument("--entity", type=str, default="BILGEM_DCS_RL")
    parser.add_argument("--run-name", type=str, default=None)
    # Logging throttle
    parser.add_argument("--table-log-every", type=int, default=TABLE_LOG_EVERY,
                        help="Flush obs table to W&B every N steps (1 = every step).")
    return parser

# ------------------------------ MAIN ------------------------------ #
def main():
    args = build_parser().parse_args()

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
                "table_log_every": int(args.table_log_every),
            },
            sync_tensorboard=True,          # keep TB sync ON
            dir=_WANDB_EPHEMERAL_DIR,       # W&B files under ephemeral dir (and env vars set)
            save_code=False,
        )
        # Define RL step axis for our metrics
        run.define_metric("train/*", step_metric="train/rl_step")

        # Put TB logs under ephemeral dir too (auto-cleaned)
        tb_dir = os.path.join(_WANDB_EPHEMERAL_DIR, "tb")

        # model = PPO(
        #     "MlpPolicy",
        #     env,
        #     n_steps=256,
        #     verbose=1,
        #     tensorboard_log=tb_dir,
        #     device=device
        # )

        model_path = r'C:\Users\fisne\OneDrive\Desktop\dcs-ml\models\ppo_harfang_RL-LLM_test06.zip'
        # ppo_harfang_v14_15-09-2025_01.zip -> Pure RL checkpoint
        model = PPO.load(model_path, env=env, device=device)
        # model.env = env
        print(f"✅ Starting training from {model_path}\n")



        # Callbacks
        wandb_callback = WandbCallback(
            gradient_save_freq=20,
            model_save_path=None,
            verbose=2
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=10_000,
            save_path=args.models_dir,
            name_prefix=args.save_name + "_checkpoint"
        )
        obs_cb = WandbFullObsLogger(table_log_every=int(args.table_log_every))

        # Train
        if args.LLM_assist:
            patch_policy_forward_with_llm(model,
                                          override_agent=harfang_agent,
                                          override_every=4,
                                          env_name="Harfang") # OVERRIDE PATCH


        model.learn(
            total_timesteps=int(args.total_steps),
            callback=[checkpoint_callback, wandb_callback, obs_cb],
            log_interval=1,
            progress_bar=True
        )

        # Save final model (NOT ephemeral)
        final_path = os.path.join(args.models_dir, args.save_name)
        model.save(final_path)
        print(f"\n✅ PPO model saved to: {final_path}")
        wandb.summary["final_model_path"] = final_path

        run.finish()
        # Clean ephemeral W&B dir
        _cleanup_dir(_WANDB_EPHEMERAL_DIR)

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
                "table_log_every": int(args.table_log_every),
            },
            sync_tensorboard=True,     # TB sync ON
            dir=_WANDB_EPHEMERAL_DIR,
            save_code=False,
        )
        # Define RL step axis for predict metrics
        run.define_metric("predict/*", step_metric="predict/rl_step")

        model_path = r'C:\Users\fisne\OneDrive\Desktop\dcs-ml\models\ppo_harfang_RL-LLM_test06.zip'
        # ppo_harfang_v14_15-09-2025_01.zip -> Pure RL checkpoint
        model = PPO.load(model_path, device=device)
        print(f"✅ Model loaded from {model_path}\n")
        print("Starting prediction...")

        global_step = 0
        start_wall = time.time()

        # Per-run full-obs table (incremental)
        pred_table = None
        pred_keys = None
        last_row = None
        LOG_EVERY = max(1, int(args.table_log_every))

        for ep in range(1, int(args.episodes) + 1):
            obs, info = env.reset()
            done, truncated = False, False
            ep_reward = 0.0
            steps = 0
            ep_start = time.time()

            while not (done or truncated):
                # Keep predict aligned with training by default
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = env.step(action)

                cur_step = int(global_step)

                # Full obs dict → table row
                sd = getattr(env.unwrapped, "state_dict", None)
                if sd is not None:
                    if pred_table is None:
                        pred_keys = sorted(list(sd.keys()))
                        pred_table = wandb.Table(
                            columns=["timestep", "episode"] + pred_keys,
                            log_mode="INCREMENTAL"
                        )
                    row = [cur_step, int(ep)] + [float(sd.get(k, np.nan)) for k in pred_keys]
                    pred_table.add_data(*row)
                    last_row = row

                    # Periodic flush
                    if (cur_step % LOG_EVERY) == 0:
                        safe_wandb_log({"predict/rl_step": cur_step, "predict/obs_full_table": pred_table})

                # Per-step scalars
                ep_reward += float(reward)
                steps += 1
                safe_wandb_log({
                    "predict/rl_step": cur_step,
                    "predict/step_reward": float(reward),
                    "predict/global_step": cur_step,
                })

                global_step += 1  # increment at end to keep cur_step consistent

            # Episode end snapshot
            if last_row is not None:
                ep_table = wandb.Table(columns=["timestep", "episode"] + pred_keys, log_mode="MUTABLE")
                ep_table.add_data(*last_row)
                safe_wandb_log({"predict/rl_step": int(global_step), "predict/obs_last_step_ep": ep_table})

            ep_time = time.time() - ep_start
            fps = steps / ep_time if ep_time > 0 else 0.0

            # Per-episode scalars
            safe_wandb_log({
                "predict/rl_step": int(global_step),
                "predict/episode": ep,
                "predict/episode_return": ep_reward,
                "predict/episode_length": steps,
                "predict/episode_success": int(info.get("success")),
                "predict/episode_fps": fps,
                "predict/mean_reward_per_step": (ep_reward / max(1, steps)),
            })
            scss = int(info.get("success"))
            print(f"Episode: {ep} | Reward: {ep_reward}.2f | Steps: {steps} | Success: {scss}")

        total_wall = time.time() - start_wall
        wandb.summary["predict/total_wall_time_sec"] = total_wall
        run.finish()
        _cleanup_dir(_WANDB_EPHEMERAL_DIR)
        print("✅ Prediction finished.")

if __name__ == "__main__":
    main()
