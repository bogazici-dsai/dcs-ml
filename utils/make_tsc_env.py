'''
@Author: WANG Maonan
@Date: 2023-09-08 17:45:54
@Description: 创建 TSC Env + Wrapper
LastEditTime: 2024-11-05 16:33:33
'''
import gymnasium as gym
import gym as gymmer
from utils.tsc_env import TSCEnvironment
from utils.tsc_wrapper import TSCEnvWrapper
from stable_baselines3.common.monitor import Monitor

def make_env(
        tls_id:str,num_seconds:int,sumo_cfg:str,use_gui:bool,
        log_file:str, env_index:int, **kwargs
        ):
    def _init() -> gym.Env:
        # TODO: Inject the env in here
        env = gymmer.make("MiniGrid-DoorKey-5x5-v0")
        for i in range(3):
            print("GYM ENV CREATED")
        # tsc_scenario = TSCEnvironment(
        #     sumo_cfg=sumo_cfg,
        #     num_seconds=num_seconds,
        #     tls_ids=[tls_id],
        #     tls_action_type='choose_next_phase',
        #     use_gui=use_gui,
        #     trip_info = './log/trip_info.xml',
        # )
        # tsc_wrapper = TSCEnvWrapper(tsc_scenario, tls_id=tls_id)
        # TODO: redesign the naming, discard Monitor if its unnecessary
        env_wrapper = TSCEnvWrapper(env, tls_id=tls_id)

        return Monitor(env_wrapper, filename=f'{log_file}/{env_index}')
    
    return _init
