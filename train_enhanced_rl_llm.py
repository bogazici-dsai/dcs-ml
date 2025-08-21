#!/usr/bin/env python3
# Enhanced RL-LLM Training Script for Harfang Combat Environment
import argparse
import os
import time
import json
import numpy as np
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda

# Import enhanced components
try:
    from env.hirl.environments.HarfangEnv_GYM_Enhanced import HarfangEnhancedEnv
    HAS_ENHANCED = True
except ImportError:
    try:
        from env.hirl.environments.HarfangEnv_GYM_ppo_v2 import HarfangEnv as HarfangEnhancedEnv
        HAS_ENHANCED = True
        print("[WARNING] Using V2 environment as Enhanced not available")
    except ImportError:
        from env.hirl.environments.HarfangEnv_GYM import HarfangEnv as HarfangEnhancedEnv
        HAS_ENHANCED = False
        print("[WARNING] Using basic environment - Enhanced/V2 not available")

from HarfangAssistant_Enhanced import HarfangTacticalAssistant
from agents.enhanced_ppo_agent import EnhancedPPOAgent, create_enhanced_ppo_config
from agents.multi_rl_trainer import MultiRLTrainer
from utils.enhanced_dataset_logger import TacticalDataLogger


def create_training_config(algorithm: str = "PPO") -> Dict[str, Any]:
    """Create optimized training configuration for Harfang combat environment"""
    
    base_config = {
        # Environment settings
        'max_episode_steps': 2000,
        'random_reset': True,
        
        # Training settings
        'total_timesteps': 1000000,
        'save_freq': 50000,
        'eval_freq': 25000,
        
        # LLM settings
        'llm_rate_hz': 10.0,
        'llm_temperature': 0.0,
        'continuous_feedback': True,
        
        # Logging settings
        'detailed_logging': True,
        'use_wandb': True,
        'log_dir': 'data/harfang_tactical_logs',
        
        # Device settings
        'device': 'auto'
    }
    
    # Algorithm-specific configurations
    if algorithm == "PPO":
        base_config.update(create_enhanced_ppo_config())
    elif algorithm == "SAC":
        base_config.update({
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'learning_starts': 10000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'ent_coef': 'auto'
        })
    elif algorithm == "TD3":
        base_config.update({
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'learning_starts': 10000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'policy_delay': 2
        })
    
    return base_config


def setup_llm_assistant(model_name: str = "llama3.1:8b", temperature: float = 0.0, 
                       rate_hz: float = 10.0, verbose: bool = True) -> HarfangTacticalAssistant:
    """Setup LLM tactical assistant with specified model"""
    
    print(f"[LLM SETUP] Initializing {model_name} with temperature {temperature}")
    
    try:
        chat = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=500,  # Longer responses for detailed tactical analysis
            top_p=0.9,
            top_k=40
        )
        llm = RunnableLambda(lambda x: chat.invoke(x))
        
        # Test LLM connection
        test_response = llm.invoke("Respond with just 'TACTICAL READY'")
        if "TACTICAL READY" in str(test_response.content):
            print(f"[LLM SETUP] {model_name} connection successful")
        else:
            print(f"[WARNING] {model_name} test response unexpected: {test_response.content}")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize {model_name}: {e}")
        raise
    
    # Initialize tactical assistant
    assistant = HarfangTacticalAssistant(
        llm=llm,
        verbose=verbose,
        max_rate_hz=rate_hz
    )
    
    print(f"[LLM SETUP] Tactical assistant ready with {rate_hz} Hz rate limit")
    return assistant


def setup_environment(max_episode_steps: int = 2000, random_reset: bool = True) -> HarfangEnhancedEnv:
    """Setup Harfang enhanced environment"""
    
    print(f"[ENV SETUP] Initializing Harfang environment")
    print(f"[ENV SETUP] Max episode steps: {max_episode_steps}")
    print(f"[ENV SETUP] Random reset: {random_reset}")
    
    try:
        env = HarfangEnhancedEnv(max_episode_steps=max_episode_steps)
        
        print(f"[ENV SETUP] Environment created successfully")
        print(f"[ENV SETUP] Observation space: {env.observation_space.shape}")
        print(f"[ENV SETUP] Action space: {env.action_space.shape}")
        
        # Test environment
        obs = env.reset()
        print(f"[ENV SETUP] Test reset successful, obs shape: {obs.shape}")
        
        return env
        
    except Exception as e:
        print(f"[ERROR] Environment setup failed: {e}")
        raise


def train_single_algorithm(algorithm: str, config: Dict[str, Any], 
                          llm_assistant: HarfangTacticalAssistant,
                          env: HarfangEnhancedEnv) -> Dict[str, Any]:
    """Train a single RL algorithm with LLM guidance"""
    
    print(f"\n{'='*60}")
    print(f"TRAINING {algorithm.upper()} WITH LLM GUIDANCE")
    print(f"{'='*60}")
    
    if algorithm == "PPO":
        # Use enhanced PPO agent
        agent = EnhancedPPOAgent(
            env=env,
            llm_assistant=llm_assistant,
            config=config,
            model_name=f"enhanced_{algorithm.lower()}",
            use_wandb=config.get('use_wandb', True),
            device=config.get('device', 'auto')
        )
        
        # Train with LLM guidance
        agent.train(
            total_timesteps=config.get('total_timesteps', 1000000),
            save_freq=config.get('save_freq', 50000),
            eval_freq=config.get('eval_freq', 25000)
        )
        
        # Evaluate trained agent
        eval_results = agent.evaluate(num_episodes=20, deterministic=True)
        
        return {
            'algorithm': algorithm,
            'config': config,
            'eval_results': eval_results,
            'model_path': f"{agent.model_dir}/{agent.model_name}_final.zip"
        }
    
    else:
        # Use multi-algorithm trainer for SAC/TD3
        trainer = MultiRLTrainer(env, llm_assistant, config)
        result = trainer.train_algorithm(
            algorithm_name=algorithm,
            total_timesteps=config.get('total_timesteps', 1000000),
            use_wandb=config.get('use_wandb', True)
        )
        return result


def main():
    """Main training script with enhanced RL-LLM integration"""
    
    parser = argparse.ArgumentParser(description='Enhanced Harfang RL-LLM Training')
    
    # Algorithm selection
    parser.add_argument('--algorithm', type=str, choices=['PPO', 'SAC', 'TD3', 'ALL'], 
                       default='PPO', help='RL algorithm to train')
    
    # LLM configuration
    parser.add_argument('--llm_model', type=str, default='llama3.1:8b',
                       help='LLM model for tactical guidance')
    parser.add_argument('--llm_rate_hz', type=float, default=10.0,
                       help='LLM call rate limit (Hz)')
    parser.add_argument('--llm_temperature', type=float, default=0.0,
                       help='LLM temperature')
    
    # Training configuration
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                       help='Total training timesteps')
    parser.add_argument('--max_episode_steps', type=int, default=2000,
                       help='Maximum steps per episode')
    
    # System configuration
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'],
                       default='auto', help='Training device')
    parser.add_argument('--use_wandb', action='store_true', default=True,
                       help='Use Weights & Biases logging')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*80)
    print("ENHANCED HARFANG RL-LLM TRAINING SYSTEM")
    print("="*80)
    print(f"Algorithm: {args.algorithm}")
    print(f"LLM Model: {args.llm_model} ({args.llm_rate_hz} Hz)")
    print(f"Training: {args.total_timesteps:,} timesteps")
    print(f"Episode Length: {args.max_episode_steps} steps")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Setup components
    print("\n[SETUP] Initializing components...")
    
    # 1. Setup LLM assistant
    llm_assistant = setup_llm_assistant(
        model_name=args.llm_model,
        temperature=args.llm_temperature,
        rate_hz=args.llm_rate_hz,
        verbose=args.verbose
    )
    
    # 2. Setup environment
    env = setup_environment(
        max_episode_steps=args.max_episode_steps,
        random_reset=True
    )
    
    # 3. Create training configuration
    config = create_training_config(args.algorithm)
    config.update({
        'total_timesteps': args.total_timesteps,
        'device': args.device,
        'use_wandb': args.use_wandb,
        'llm_rate_hz': args.llm_rate_hz,
        'llm_temperature': args.llm_temperature
    })
    
    print(f"[SETUP] Configuration created for {args.algorithm}")
    
    # 4. Start training
    if args.algorithm == "ALL":
        print("\n[TRAINING] Comparing all algorithms...")
        trainer = MultiRLTrainer(env, llm_assistant, config)
        results = trainer.compare_algorithms(
            algorithms=['PPO', 'SAC', 'TD3'],
            total_timesteps=args.total_timesteps
        )
        
        print(f"\n[COMPLETE] All algorithms trained. Results saved in models/multi_rl_comparison/")
        
    else:
        print(f"\n[TRAINING] Training {args.algorithm}...")
        result = train_single_algorithm(args.algorithm, config, llm_assistant, env)
        
        print(f"\n[COMPLETE] {args.algorithm} training finished!")
        print(f"Model saved to: {result.get('model_path', 'unknown')}")
        if 'eval_results' in result:
            eval_results = result['eval_results']
            print(f"Evaluation: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    
    # Cleanup
    env.close()
    print("\n[COMPLETE] Training session finished!")


if __name__ == "__main__":
    main()
