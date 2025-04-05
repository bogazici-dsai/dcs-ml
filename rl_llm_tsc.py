import os
os.environ['SUMO_HOME'] = '/opt/miniconda3/envs/llmrl'
os.environ['PATH'] = f"/opt/miniconda3/envs/llmrl/bin:{os.environ['PATH']}"

import torch
from loguru import logger

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda

from tshub.utils.format_dict import dict_to_str
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from TSCEnvironment.tsc_env import TSCEnvironment
from TSCEnvironment.tsc_env_wrapper import TSCEnvWrapper
from TSCAssistant.tsc_assistant import TSCAgent

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from utils.make_tsc_env import make_env
from utils.readConfig import read_config

# ============
# Load config.yaml for LLM model settings
# ============
config = read_config()
llm_model_name = config["LLM_MODEL"]
llm_temperature = config.get("LLM_TEMPERATURE", 0.0)

chat = ChatOllama(
    model=llm_model_name,
    temperature=llm_temperature
)
llm = RunnableLambda(lambda x: chat.invoke(x))  # wrap in LangChain interface


# ============
# Logger and path init
# ============
path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))


if __name__ == '__main__':
    # ============
    # SUMO Config and Env Setup
    # ============
    sumo_cfg = path_convert("./TSCScenario/SumoNets/train_four_345/env/train_four_345.sumocfg")
    trip_info = path_convert('./Result/LLM.tripinfo.xml')

    params = {
        'tls_id': 'J1',
        'num_seconds': 300,
        'sumo_cfg': sumo_cfg,
        'use_gui': True,
        'log_file': './log_test/',
        'trip_info': trip_info,
    }

    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(1)])
    env = VecNormalize.load(load_path=path_convert('./models/last_vec_normalize.pkl'), venv=env)
    env.training = False
    env.norm_reward = False

    # ============
    # Load trained RL model
    # ============
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model_path = path_convert('./models/last_rl_model.zip')
    model = PPO.load(model_path, env=env, device=device)

    # ============
    # Initialize TSCAgent using local Ollama
    # ============
    tsc_agent = TSCAgent(llm=llm, verbose=True)

    # ============
    # Run test loop
    # ============
    dones = False
    sim_step = 0
    obs = env.reset()

    while not dones:
        action, _state = model.predict(obs, deterministic=True)

        if sim_step > 4:
            action = tsc_agent.agent_run(
                sim_step=sim_step,
                action=action,
                obs=obs,
                infos=infos  # previous step’s infos
            ),

        obs, rewards, dones, infos = env.step(action)
        sim_step += 1

    print('*********** Total Rewards ************', rewards)
    env.close()