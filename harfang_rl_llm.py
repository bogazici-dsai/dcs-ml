import argparse
import os
import time
from typing import Optional

import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda

from env.hirl.environments.HarfangEnv_GYM import HarfangEnv as HarfangEnvV1  # base
try:
    from env.hirl.environments.HarfangEnv_GYM_ppo_v2 import HarfangEnv as HarfangEnvV2  # rich info
    HAS_V2 = True
except Exception:
    HAS_V2 = False

from harfang_assistant_new.harfang_assistant import HarfangAssistant
from utils.dataset_logger import CsvStepLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_model', type=str, default='llama3.1:8b')
    parser.add_argument('--llm_rate_hz', type=float, default=10.0)
    parser.add_argument('--env_version', type=str, choices=['v1', 'v2', 'auto'], default='auto')
    parser.add_argument('--max_steps', type=int, default=1500)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--csv_dir', type=str, default='data/harfang_rl_llm_logs')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # LLM setup
    chat = ChatOllama(model=args.llm_model, temperature=0.0)
    llm = RunnableLambda(lambda x: chat.invoke(x))
    assistant = HarfangAssistant(llm=llm, verbose=True, max_rate_hz=args.llm_rate_hz)

    # Env selection
    env: Optional[object] = None
    use_v2 = (args.env_version == 'v2') or (args.env_version == 'auto' and HAS_V2)
    if use_v2:
        env = HarfangEnvV2(max_episode_steps=args.max_steps)
    else:
        env = HarfangEnvV1()

    # Logger
    logger = CsvStepLogger(args.csv_dir)

    rng = np.random.default_rng(args.seed)
    prev_state = None
    prev_action = None
    lock_duration = 0

    for ep in range(args.episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        prev_state = None
        prev_action = None
        lock_duration = 0

        for step in range(args.max_steps):
            if done:
                break

            # Placeholder agent policy: random action for now
            action = env.action_space.sample()

            # Step environment; support both API variants
            step_result = env.step(action)
            if len(step_result) == 5:
                n_state, base_reward, done, info, step_success = step_result
            else:
                # v2 returns info with components
                n_state, base_reward, done, info = step_result
                step_success = info.get('step_success', 0) if isinstance(info, dict) else 0

            # Track lock duration (if state[7] > 0 indicates locked in v1 packing)
            try:
                locked_flag = 1 if float(n_state[7]) > 0 else -1
                if locked_flag > 0:
                    lock_duration += 1
                else:
                    lock_duration = 0
            except Exception:
                pass

            # Build features and request shaping every step (continuous), but LLM may rate-limit internally
            features = assistant.extract_features(
                state=n_state,
                prev_state=prev_state,
                action=action,
                info=info if isinstance(info, dict) else None,
                lock_duration=lock_duration,
                prev_action=prev_action,
            )
            shaping_delta, llm_raw = assistant.request_shaping(features)

            final_reward = float(base_reward) + float(shaping_delta)
            total_reward += final_reward

            # CSV log
            row = {
                'episode': ep,
                'step': step,
                'base_reward': float(base_reward),
                'shaping_delta': float(shaping_delta),
                'final_reward': final_reward,
                'done': int(bool(done)),
                'action_pitch': float(action[0]),
                'action_roll': float(action[1]),
                'action_yaw': float(action[2]),
                'action_fire': float(action[3]),
                'dx': float(n_state[0]) if len(n_state) > 0 else np.nan,
                'dy': float(n_state[1]) if len(n_state) > 1 else np.nan,
                'dz': float(n_state[2]) if len(n_state) > 2 else np.nan,
                'target_angle': float(n_state[6]) if len(n_state) > 6 else np.nan,
                'locked': float(n_state[7]) if len(n_state) > 7 else np.nan,
                'missile1_state': float(n_state[8]) if len(n_state) > 8 else np.nan,
                'enemy_health': float(n_state[12]) if len(n_state) > 12 else np.nan,
                'llm_json': str(llm_raw),
                'step_success': int(step_success),
            }
            logger.log(row)

            # Advance
            prev_state = n_state
            prev_action = action
            state = n_state

        print(f"Episode {ep+1}/{args.episodes} total_reward={total_reward:.2f}")

    logger.close()


if __name__ == '__main__':
    main()


