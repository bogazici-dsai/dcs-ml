#!/usr/bin/env python3
# Enhanced Harfang RL-LLM Integration with Multi-Model Support and Advanced Features
import argparse
import os
import time
import json
import numpy as np
from typing import Optional, Dict, Any

# Enhanced LLM support
from llm.multi_llm_manager import MultiLLMManager, interactive_model_selection
from llm.multi_stage_tactical_assistant import EnhancedTacticalAssistant

# Enhanced RL training
from agents.enhanced_ppo_agent import EnhancedPPOAgent, create_enhanced_ppo_config
from agents.multi_rl_trainer import MultiRLTrainer

# Environment and logging
try:
    from env.hirl.environments.HarfangEnv_GYM_Enhanced import HarfangEnhancedEnv
    HAS_ENHANCED = True
    ENV_TYPE = "enhanced"
except ImportError:
    try:
        from env.hirl.environments.HarfangEnv_GYM_ppo_v2 import HarfangEnv as HarfangEnhancedEnv
        HAS_ENHANCED = True
        ENV_TYPE = "v2"
        print("[WARNING] Using V2 environment as Enhanced not available")
    except ImportError:
        try:
            from env.hirl.environments.HarfangEnv_GYM import HarfangEnv as HarfangEnhancedEnv
            HAS_ENHANCED = False
            ENV_TYPE = "basic"
            print("[WARNING] Using basic environment")
        except ImportError:
            from env.mock_harfang_env import MockHarfangEnhancedEnv as HarfangEnhancedEnv
            HAS_ENHANCED = False
            ENV_TYPE = "mock"
            print("[INFO] Using mock Harfang environment for testing")

from utils.enhanced_dataset_logger import TacticalDataLogger


def main():
    """Enhanced main function with multi-LLM support and advanced training"""
    
    parser = argparse.ArgumentParser(description='Enhanced Harfang RL-LLM with Multi-Model Support')
    
    # LLM Configuration
    parser.add_argument('--llm_model', type=str, default='gemma3:4b',
                       help='LLM model (gemma3:4b default, auto for interactive selection, or specific model ID)')
    parser.add_argument('--llm_temperature', type=float, default=0.0,
                       help='LLM temperature for response generation')
    parser.add_argument('--llm_rate_hz', type=float, default=10.0,
                       help='Maximum LLM call rate (calls per second)')
    parser.add_argument('--multi_stage_reasoning', action='store_true', default=True,
                       help='Enable multi-stage reasoning (Strategic/Tactical/Execution)')
    
    # RL Algorithm Configuration
    parser.add_argument('--algorithm', type=str, choices=['PPO', 'SAC', 'TD3', 'ALL'], 
                       default='PPO', help='RL algorithm to use (PPO default)')
    parser.add_argument('--compare_algorithms', action='store_true', default=False,
                       help='Compare multiple RL algorithms (PPO, SAC, TD3)')
    parser.add_argument('--algorithm_variants', action='store_true', default=False,
                       help='Train multiple variants of the selected algorithm')
    
    # Training Configuration
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'demo', 'benchmark'],
                       default='train', help='Operation mode')
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                       help='Total training timesteps')
    parser.add_argument('--max_episode_steps', type=int, default=2000,
                       help='Maximum steps per episode')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes (for evaluate/demo modes)')
    
    # Environment Configuration
    parser.add_argument('--random_reset', action='store_true', default=True,
                       help='Use randomized initial positions')
    parser.add_argument('--mission_type', type=str, default='air_superiority',
                       choices=['air_superiority', 'intercept', 'escort', 'cas'],
                       help='Mission type for strategic context')
    
    # System Configuration
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'],
                       default='auto', help='Training device')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Enable Weights & Biases logging')
    parser.add_argument('--no_wandb', action='store_true', default=False,
                       help='Explicitly disable Weights & Biases logging')
    parser.add_argument('--detailed_logging', action='store_true', default=True,
                       help='Enable detailed tactical logging')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose output')
    
    # Advanced Features
    parser.add_argument('--benchmark_llms', action='store_true',
                       help='Benchmark available LLM models')
    parser.add_argument('--interactive_model_selection', action='store_true',
                       help='Interactive LLM model selection')
    
    args = parser.parse_args()
    
    # Print header
    print("="*80)
    print("ENHANCED HARFANG RL-LLM TACTICAL TRAINING SYSTEM")
    print("="*80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Algorithm: {args.algorithm}")
    print(f"LLM Model: {args.llm_model}")
    print(f"Multi-Stage Reasoning: {args.multi_stage_reasoning}")
    print(f"Mission Type: {args.mission_type}")
    print("="*80)
    
    # Initialize Multi-LLM Manager
    print("\n[SETUP] Initializing Multi-LLM Manager...")
    llm_manager = MultiLLMManager(verbose=args.verbose)
    
    # Handle special modes first
    if args.benchmark_llms:
        print("\n[BENCHMARK] Benchmarking available LLM models...")
        available_models = [model_id for model_id, _ in llm_manager.supported_models.items()]
        benchmark_results = llm_manager.benchmark_models(available_models[:5])  # Test top 5
        return
    
    # LLM Model Selection
    if args.llm_model == 'auto' or args.interactive_model_selection:
        print("\n[SELECTION] Interactive LLM model selection...")
        selected_model = interactive_model_selection()
        if selected_model is None:
            print("[ERROR] No model selected. Exiting.")
            return
    else:
        selected_model = args.llm_model
        if selected_model not in llm_manager.supported_models:
            print(f"[ERROR] Model {selected_model} not supported")
            print("Available models:", list(llm_manager.supported_models.keys()))
            return
    
    print(f"\n[LLM] Using model: {llm_manager.supported_models[selected_model].name}")
    
    # Initialize LLM Tactical Assistant
    print("[SETUP] Initializing Enhanced Tactical Assistant...")
    try:
        base_llm = llm_manager.initialize_model(
            selected_model, 
            temperature=args.llm_temperature,
            max_rate_hz=args.llm_rate_hz
        )
        
        if base_llm is None:
            print("[ERROR] Failed to initialize LLM")
            return
        
        # Create enhanced tactical assistant
        tactical_assistant = EnhancedTacticalAssistant(
            llm=base_llm,
            verbose=args.verbose,
            max_rate_hz=args.llm_rate_hz,
            use_multi_stage=args.multi_stage_reasoning
        )
        
        # Update mission context
        tactical_assistant.update_mission_context(
            mission_type=args.mission_type,
            threat_environment="medium",
            fuel_state="normal",
            ammunition_state="full"
        )
        
        print(f"[SETUP] Enhanced Tactical Assistant ready")
        
    except Exception as e:
        print(f"[ERROR] LLM setup failed: {e}")
        return
    
    # Initialize Environment
    print("[SETUP] Initializing Harfang Environment...")
    try:
        env = HarfangEnhancedEnv(max_episode_steps=args.max_episode_steps)
        print(f"[SETUP] Environment ready - Obs: {env.observation_space.shape}, Action: {env.action_space.shape}")
    except Exception as e:
        print(f"[ERROR] Environment setup failed: {e}")
        return
    
    # Initialize Enhanced Logging
    if args.detailed_logging:
        logger = TacticalDataLogger(
            out_dir="data/enhanced_harfang_logs",
            filename_prefix=f"enhanced_{args.algorithm.lower()}_{selected_model.replace(':', '_')}",
            create_separate_files=True
        )
        print("[SETUP] Enhanced tactical logging enabled")
    else:
        logger = None
    
    # Execute based on mode
    if args.mode == 'train':
        print(f"\n[TRAINING] Starting {args.algorithm} training with {selected_model}")
        
        if args.compare_algorithms or args.algorithm == 'ALL':
            # Multi-algorithm comparison
            config = create_enhanced_ppo_config()
            # Handle WandB configuration
            use_wandb = args.use_wandb and not args.no_wandb
            
            config.update({
                'total_timesteps': args.total_timesteps,
                'device': args.device,
                'use_wandb': use_wandb
            })
            
            trainer = MultiRLTrainer(env, tactical_assistant, config)
            results = trainer.compare_algorithms(
                algorithms=['PPO', 'SAC', 'TD3'],
                total_timesteps=args.total_timesteps
            )
            
            print(f"\n[COMPLETE] Algorithm comparison finished")
            
        else:
            # Single algorithm training
            if args.algorithm == 'PPO':
                config = create_enhanced_ppo_config()
                # Handle WandB configuration
                use_wandb = args.use_wandb and not args.no_wandb
                
                config.update({
                    'total_timesteps': args.total_timesteps,
                    'device': args.device,
                    'use_wandb': use_wandb
                })
                
                agent = EnhancedPPOAgent(
                    env=env,
                    llm_assistant=tactical_assistant,
                    config=config,
                    model_name=f"enhanced_ppo_{selected_model.replace(':', '_')}",
                    use_wandb=use_wandb,
                    device=args.device
                )
                
                agent.train(
                    total_timesteps=args.total_timesteps,
                    save_freq=50000,
                    eval_freq=25000
                )
                
                print(f"[COMPLETE] PPO training finished")
                
            else:
                # Use multi-trainer for SAC/TD3
                config = {'total_timesteps': args.total_timesteps, 'device': args.device}
                trainer = MultiRLTrainer(env, tactical_assistant, config)
                result = trainer.train_algorithm(args.algorithm, args.total_timesteps)
                
                print(f"[COMPLETE] {args.algorithm} training finished")
    
    elif args.mode == 'evaluate':
        print(f"\n[EVALUATION] Evaluating trained models...")
        # TODO: Implement model evaluation
        print("[INFO] Evaluation mode not yet implemented")
    
    elif args.mode == 'demo':
        print(f"\n[DEMO] Running demonstration with LLM guidance...")
        # TODO: Implement interactive demo
        print("[INFO] Demo mode not yet implemented")
    
    elif args.mode == 'benchmark':
        print(f"\n[BENCHMARK] Benchmarking system performance...")
        # TODO: Implement system benchmarking
        print("[INFO] Benchmark mode not yet implemented")
    
    # Cleanup
    if logger:
        logger.close()
    env.close()
    
    print(f"\n[COMPLETE] Enhanced Harfang RL-LLM session finished!")
    print("="*80)


if __name__ == "__main__":
    main()
