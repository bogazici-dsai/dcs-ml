'''
@Author: WANG Maonan
@Author: PangAoyu
@Description: Test a pre-trained RL Agent in traffic signal control environment
'''

import torch
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from utils.make_tsc_env import make_env

# Resolve relative paths
path_convert = get_abs_path(__file__)
logger.remove()

if __name__ == '__main__':
    ####################
    # Initialize Environment
    ####################
    sumo_cfg = path_convert("./TSCScenario/SumoNets/train_four_345/env/train_four_345.sumocfg")
    params = {
        'tls_id': 'J1',
        'num_seconds': 1600,
        'sumo_cfg': sumo_cfg,
        'use_gui': True,
        'log_file': path_convert('./log/'),
    }

    # Create vectorized environment with normalization
    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(1)])
    env = VecNormalize.load(load_path=path_convert('./models/last_vec_normalize.pkl'), venv=env)
    env.training = False  # Disable training during test
    env.norm_reward = False  # Do not normalize rewards during test

    ####################
    # Load Trained RL Model
    ####################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = path_convert('./models/last_rl_model.zip')
    model = PPO.load(model_path, env=env, device=device)

    ####################
    # Run Inference
    ####################
    obs = env.reset()
    dones = False
    total_reward = 0

    while not dones:
        action, _ = model.predict(obs, deterministic=True)
        print('action:', action)
        print('observation:', obs)
        obs, rewards, dones, infos = env.step(action)
        total_reward += rewards

    env.close()
    print(f'Total Reward: {total_reward}')