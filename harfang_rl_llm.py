import argparse
import os
import time
import json
import numpy as np
from typing import Optional, Dict, Any, List
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda

# Enhanced environment imports
from env.hirl.environments.HarfangEnv_GYM import HarfangEnv as HarfangEnvV1  # base
try:
    from env.hirl.environments.HarfangEnv_GYM_ppo_v2 import HarfangEnv as HarfangEnvV2  # rich info
    HAS_V2 = True
except Exception:
    HAS_V2 = False

try:
    from env.hirl.environments.HarfangEnv_GYM_Enhanced import HarfangEnhancedEnv  # comprehensive tactical
    HAS_ENHANCED = True
except Exception:
    HAS_ENHANCED = False
    print("[WARNING] Enhanced environment not available, falling back to v2")

# Enhanced assistant and logging
from HarfangAssistant_Enhanced import HarfangTacticalAssistant
from utils.enhanced_dataset_logger import TacticalDataLogger, CsvStepLogger


def main():
    parser = argparse.ArgumentParser(description='Enhanced Harfang RL-LLM Integration with Tactical Analysis')
    
    # LLM Configuration
    parser.add_argument('--llm_model', type=str, default='llama3.1:8b', 
                       help='LLM model to use for tactical guidance')
    parser.add_argument('--llm_rate_hz', type=float, default=10.0,
                       help='Maximum LLM call rate (calls per second)')
    parser.add_argument('--llm_temperature', type=float, default=0.0,
                       help='LLM temperature for response generation')
    
    # Environment Configuration
    parser.add_argument('--env_version', type=str, choices=['v1', 'v2', 'enhanced', 'auto'], 
                       default='auto', help='Environment version to use')
    parser.add_argument('--max_steps', type=int, default=1500,
                       help='Maximum steps per episode')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--random_reset', action='store_true',
                       help='Use random initial positions')
    
    # Agent Configuration
    parser.add_argument('--agent_type', type=str, choices=['random', 'rule', 'trained'], 
                       default='random', help='Agent type to use')
    parser.add_argument('--agent_model_path', type=str, default=None,
                       help='Path to trained agent model (if using trained agent)')
    
    # Logging Configuration
    parser.add_argument('--csv_dir', type=str, default='data/harfang_tactical_logs',
                       help='Directory for CSV log files')
    parser.add_argument('--detailed_logging', action='store_true',
                       help='Enable detailed tactical logging with separate files')
    parser.add_argument('--log_prefix', type=str, default='harfang_enhanced',
                       help='Prefix for log files')
    
    # Experimental Features
    parser.add_argument('--continuous_feedback', action='store_true', default=True,
                       help='Enable continuous LLM feedback (every step)')
    parser.add_argument('--action_space_evolution', action='store_true',
                       help='Enable experimental action space evolution')
    parser.add_argument('--tactical_analysis', action='store_true', default=True,
                       help='Enable comprehensive tactical analysis')
    
    # System Configuration
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("ENHANCED HARFANG RL-LLM TACTICAL INTEGRATION")
    print("=" * 60)
    print(f"LLM Model: {args.llm_model} (Rate: {args.llm_rate_hz} Hz)")
    print(f"Environment: {args.env_version} | Episodes: {args.episodes} | Max Steps: {args.max_steps}")
    print(f"Agent: {args.agent_type} | Continuous Feedback: {args.continuous_feedback}")
    print(f"Logging: {args.csv_dir} (Detailed: {args.detailed_logging})")
    print("=" * 60)

    # Enhanced LLM setup with tactical assistant
    print(f"[INIT] Initializing LLM: {args.llm_model}")
    chat = ChatOllama(model=args.llm_model, temperature=args.llm_temperature)
    llm = RunnableLambda(lambda x: chat.invoke(x))
    
    # Use enhanced tactical assistant
    assistant = HarfangTacticalAssistant(
        llm=llm, 
        verbose=args.verbose, 
        max_rate_hz=args.llm_rate_hz
    )
    print(f"[INIT] Tactical assistant initialized with {args.llm_rate_hz} Hz rate limit")

    # Enhanced environment selection with priority: Enhanced > V2 > V1
    env: Optional[object] = None
    env_name = ""
    
    if args.env_version == 'enhanced' or (args.env_version == 'auto' and HAS_ENHANCED):
        env = HarfangEnhancedEnv(max_episode_steps=args.max_steps)
        env_name = "Enhanced (Comprehensive Tactical)"
        print(f"[INIT] Using Enhanced Harfang environment with tactical analysis")
    elif args.env_version == 'v2' or (args.env_version == 'auto' and HAS_V2):
        env = HarfangEnvV2(max_episode_steps=args.max_steps)
        env_name = "V2 (Improved Rewards)"
        print(f"[INIT] Using Harfang V2 environment")
    else:
        env = HarfangEnvV1()
        env_name = "V1 (Basic)"
        print(f"[INIT] Using basic Harfang V1 environment")
    
    print(f"[INIT] Environment: {env_name}")
    print(f"[INIT] Observation space: {env.observation_space.shape}")
    print(f"[INIT] Action space: {env.action_space.shape}")

    # Enhanced tactical logger
    if args.detailed_logging:
        logger = TacticalDataLogger(
            out_dir=args.csv_dir,
            filename_prefix=args.log_prefix,
            create_separate_files=True
        )
        print(f"[INIT] Detailed tactical logging enabled: {args.csv_dir}")
    else:
        logger = TacticalDataLogger(
            out_dir=args.csv_dir,
            filename_prefix=args.log_prefix,
            create_separate_files=False
        )
        print(f"[INIT] Basic logging enabled: {args.csv_dir}")

    # Initialize agent and tracking
    rng = np.random.default_rng(args.seed)
    print(f"[INIT] Random seed set to {args.seed}")
    np.random.seed(args.seed)
    
    # Agent setup (currently using random, but extensible for future trained agents)
    if args.agent_type == 'random':
        agent = RandomAgent(env.action_space)
        print(f"[INIT] Using random agent for exploration")
    else:
        agent = None
        print(f"[INIT] Agent type '{args.agent_type}' not implemented, using random fallback")
    
    # Episode tracking
    episode_results = []
    total_llm_calls = 0
    total_steps = 0
    
    print(f"\n[START] Beginning {args.episodes} episodes with {env_name} environment")
    print("=" * 60)

    for ep in range(args.episodes):
        print(f"\n[EPISODE {ep+1}/{args.episodes}] Starting...")
        
        # Reset environment (with optional randomization)
        if args.random_reset and hasattr(env, 'random_reset'):
            state = env.random_reset()
        else:
            state = env.reset()
        
        # Episode tracking variables
        done = False
        total_reward = 0.0
        base_reward_sum = 0.0
        llm_shaping_sum = 0.0
        prev_state = None
        prev_action = None
        lock_duration = 0
        episode_start_time = time.time()
        
        # Episode-level metrics tracking
        episode_metrics = {
            'shots_fired': 0,
            'shots_successful': 0,
            'max_lock_duration': 0,
            'total_lock_time': 0,
            'distances': [],
            'threat_levels': [],
            'g_forces': [],
            'altitude_violations': 0,
            'time_in_envelope': 0
        }

        for step in range(args.max_steps):
            if done:
                break

            # Agent action selection
            if agent:
                action = agent.get_action(state)
            else:
                action = env.action_space.sample()  # Fallback to random

            # Step environment with enhanced error handling
            try:
                step_result = env.step(action)
                if len(step_result) == 5:
                    n_state, base_reward, done, info, step_success = step_result
                else:
                    # Enhanced/v2 environments return 4-tuple
                    n_state, base_reward, done, info = step_result
                    step_success = info.get('step_success', 0) if isinstance(info, dict) else 0
            except Exception as e:
                print(f"[ERROR] Environment step failed: {e}")
                break

            # Enhanced lock duration tracking
            try:
                locked_flag = 1 if float(n_state[7]) > 0 else -1
                if locked_flag > 0:
                    lock_duration += 1
                    episode_metrics['total_lock_time'] += 1
                    episode_metrics['max_lock_duration'] = max(episode_metrics['max_lock_duration'], lock_duration)
                else:
                    lock_duration = 0
            except Exception:
                locked_flag = -1

            # Continuous LLM feedback with enhanced features
            llm_start_time = time.time()
            try:
                features = assistant.extract_features(
                    state=n_state,
                    prev_state=prev_state,
                    action=action,
                    info=info if isinstance(info, dict) else {},
                    lock_duration=lock_duration,
                    prev_action=prev_action,
                )
                shaping_delta, llm_raw = assistant.request_shaping(features)
                total_llm_calls += 1
            except Exception as e:
                print(f"[WARNING] LLM guidance failed: {e}")
                features = {}
                shaping_delta = 0.0
                llm_raw = {'error': str(e), 'critique': 'llm_error'}
            
            llm_time_ms = (time.time() - llm_start_time) * 1000

            # Calculate final reward and update totals
            final_reward = float(base_reward) + float(shaping_delta)
            total_reward += final_reward
            base_reward_sum += float(base_reward)
            llm_shaping_sum += float(shaping_delta)
            total_steps += 1
            
            # Update episode metrics
            if isinstance(info, dict):
                episode_metrics['distances'].append(info.get('distance', 0))
                episode_metrics['threat_levels'].append(features.get('threat_level', 0))
                episode_metrics['g_forces'].append(features.get('g_force', 0))
                episode_metrics['altitude_violations'] += info.get('altitude_violations', 0)
                
                if step_success == 1:
                    episode_metrics['shots_successful'] += 1
                if action[3] > 0:  # Fire action
                    episode_metrics['shots_fired'] += 1
                if features.get('in_firing_envelope', False):
                    episode_metrics['time_in_envelope'] += 1

            # Enhanced logging with comprehensive tactical data
            try:
                logger.log_step(
                    episode=ep,
                    step=step,
                    state=n_state,
                    action=action,
                    base_reward=base_reward,
                    shaping_delta=shaping_delta,
                    done=done,
                    info=info if isinstance(info, dict) else {},
                    features=features,
                    llm_response=llm_raw,
                    llm_time_ms=llm_time_ms
                )
            except Exception as e:
                print(f"[WARNING] Logging failed: {e}")
                # Fallback to basic logging
                basic_row = {
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
                # Use old logger interface as fallback
                if hasattr(logger, 'log'):
                    logger.log(basic_row)

            # Advance state
            prev_state = n_state
            prev_action = action
            state = n_state
            
            # Progress reporting for long episodes
            if step % 500 == 0 and step > 0:
                print(f"  Step {step}: Reward={total_reward:.2f}, Distance={features.get('distance', 0):.0f}m, Locked={locked_flag > 0}")

        # Episode completion
        episode_time = time.time() - episode_start_time
        
        # Calculate episode-level metrics
        episode_data = {
            'length': step + 1,
            'total_reward': total_reward,
            'base_reward_sum': base_reward_sum,
            'llm_shaping_sum': llm_shaping_sum,
            'shots_fired': episode_metrics['shots_fired'],
            'shots_successful': episode_metrics['shots_successful'],
            'shot_accuracy': episode_metrics['shots_successful'] / max(episode_metrics['shots_fired'], 1),
            'victory': info.get('success', False) if isinstance(info, dict) else False,
            'max_lock_duration': episode_metrics['max_lock_duration'],
            'total_lock_time': episode_metrics['total_lock_time'],
            'lock_percentage': episode_metrics['total_lock_time'] / max(step + 1, 1) * 100,
            'avg_distance': np.mean(episode_metrics['distances']) if episode_metrics['distances'] else 0,
            'min_distance': np.min(episode_metrics['distances']) if episode_metrics['distances'] else 0,
            'time_in_envelope': episode_metrics['time_in_envelope'],
            'altitude_violations': episode_metrics['altitude_violations'],
            'avg_threat_level': np.mean(episode_metrics['threat_levels']) if episode_metrics['threat_levels'] else 0,
            'max_g_force': np.max(episode_metrics['g_forces']) if episode_metrics['g_forces'] else 0,
            'tactical_efficiency': total_reward / max(step + 1, 1)  # Reward per step
        }
        
        # Log episode metrics
        if hasattr(logger, 'log_episode_metrics'):
            logger.log_episode_metrics(ep, episode_data)
        
        episode_results.append(episode_data)
        
        # Episode summary
        victory_status = "VICTORY" if episode_data['victory'] else "Defeat"
        print(f"[EPISODE {ep+1}] {victory_status} | Reward: {total_reward:.2f} | Steps: {step+1} | Time: {episode_time:.1f}s")
        print(f"  Lock: {episode_data['lock_percentage']:.1f}% | Shots: {episode_metrics['shots_fired']} ({episode_data['shot_accuracy']:.1%} acc) | Min Dist: {episode_data['min_distance']:.0f}m")
        
        if args.verbose and ep < args.episodes - 1:
            tactical_summary = assistant.get_tactical_summary() if hasattr(assistant, 'get_tactical_summary') else {}
            print(f"  LLM: {tactical_summary.get('total_guidance_calls', 0)} calls | Rate Limited: {tactical_summary.get('rate_limit_active', False)}")

    # Final session summary
    total_time = time.time() - episode_start_time if 'episode_start_time' in locals() else 0
    
    print("\n" + "=" * 60)
    print("SESSION COMPLETE - TACTICAL SUMMARY")
    print("=" * 60)
    
    if episode_results:
        avg_reward = np.mean([ep['total_reward'] for ep in episode_results])
        avg_length = np.mean([ep['length'] for ep in episode_results])
        victory_rate = np.mean([ep['victory'] for ep in episode_results]) * 100
        avg_accuracy = np.mean([ep['shot_accuracy'] for ep in episode_results]) * 100
        
        print(f"Episodes: {args.episodes} | Total Steps: {total_steps} | LLM Calls: {total_llm_calls}")
        print(f"Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f} steps")
        print(f"Victory Rate: {victory_rate:.1f}% | Shot Accuracy: {avg_accuracy:.1f}%")
        print(f"Environment: {env_name} | Agent: {args.agent_type}")
        
        # Tactical analysis summary
        if hasattr(assistant, 'get_tactical_summary'):
            tactical_summary = assistant.get_tactical_summary()
            print(f"Tactical Guidance: {tactical_summary.get('total_guidance_calls', 0)} calls")
    
    # Close logger and get session summary
    if hasattr(logger, 'get_session_summary'):
        session_summary = logger.get_session_summary()
        print(f"\nSession ID: {session_summary.get('session_id', 'unknown')}")
        print(f"Data logged to: {args.csv_dir}")
    
    logger.close()
    print("\nAll files saved. Session complete!")
    print("=" * 60)


class RandomAgent:
    """Simple random agent for testing"""
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, state):
        return self.action_space.sample()


if __name__ == '__main__':
    main()


