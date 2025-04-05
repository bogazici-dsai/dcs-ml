'''
@Author: WANG Maonan
@Author: PangAoyu
@Date: 2023-09-08
@Description: Train a traffic signal control (TSC) model using Stable Baselines3.
- State: Last step occupancy for each movement
- Action: Choose next phase
- Reward: Total waiting time
@LastEditTime: 2024-11-05
'''

import os
import torch
from loguru import logger

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from utils.make_tsc_env import make_env
from utils.sb3_utils import VecNormalizeCallback, linear_schedule
from utils.custom_models import CustomModel
from utils import scnn  # Custom spatial CNN feature extractor

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

# Initialize logging
path_convert = get_abs_path(__file__)
logger.remove()
set_logger(path_convert('./'), terminal_log_level="INFO")

if __name__ == '__main__':
    # ===== Paths =====
    log_path = path_convert('./log/')
    model_path = path_convert('./models/')
    tensorboard_path = path_convert('./tensorboard/')

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)

    # ===== Environment Setup =====
    sumo_cfg = path_convert("./TSCScenario/SumoNets/train_four_345/env/train_four_345.sumocfg")
    env_params = {
        'tls_id': 'J1',
        'num_seconds': 3600,
        'sumo_cfg': sumo_cfg,
        'use_gui': False,
        'log_file': log_path,
    }
    env = SubprocVecEnv([make_env(env_index=f'{i}', **env_params) for i in range(5)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    # ===== Callbacks =====
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=model_path,
    )
    vec_normalize_callback = VecNormalizeCallback(
        save_freq=10_000,
        save_path=model_path,
    )
    callbacks = CallbackList([checkpoint_callback, vec_normalize_callback])

    # ===== PPO Model Setup =====
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # macOS Apple Silicon support
    policy_kwargs = dict(
        features_extractor_class=scnn.SCNN,  # or CustomModel
        features_extractor_kwargs=dict(features_dim=32),
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=5000,
        n_epochs=10,
        learning_rate=linear_schedule(5e-4),
        verbose=True,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_path,
        device=device
    )

    # ===== Training =====
    model.learn(total_timesteps=300_000, tb_log_name='J1', callback=callbacks)

    # ===== Save final model and environment =====
    env.save(f'{model_path}/last_vec_normalize.pkl')
    model.save(f'{model_path}/last_rl_model.zip')
    print("Training completed and model saved.")

    env.close()