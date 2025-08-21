# Multi-Algorithm RL Trainer for Harfang Combat Environment
import os
import numpy as np
import torch
import wandb
from typing import Dict, Any, Optional, Union
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
import time


class MultiRLTrainer:
    """
    Multi-algorithm RL trainer supporting PPO, SAC, and TD3 for Harfang combat environment.
    Allows comparison of different RL algorithms with LLM guidance.
    """
    
    def __init__(self, env, llm_assistant, base_config: Dict[str, Any]):
        """
        Initialize multi-algorithm trainer
        
        Args:
            env: HarfangEnhancedEnv with 25-dimensional state space
            llm_assistant: HarfangTacticalAssistant for guidance
            base_config: Base configuration for all algorithms
        """
        self.env = env
        self.llm_assistant = llm_assistant
        self.base_config = base_config
        
        # Device selection
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        print(f"[MULTI RL] Using device: {self.device}")
        
        # Setup directories
        self.log_dir = "logs/multi_rl_comparison"
        self.model_dir = "models/multi_rl_comparison"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Algorithm configurations
        self.algorithm_configs = self._create_algorithm_configs()
        self.trained_models = {}
        
    def _create_algorithm_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create optimized configurations for each RL algorithm"""
        
        # Enhanced network architecture for 25D state space
        policy_kwargs_large = {
            "net_arch": {
                "pi": [512, 256, 128],
                "vf": [512, 256, 128]
            },
            "activation_fn": torch.nn.ReLU,
            "ortho_init": True
        }
        
        # SAC/TD3 network architecture (actor-critic)
        policy_kwargs_continuous = {
            "net_arch": [512, 256, 128],
            "activation_fn": torch.nn.ReLU
        }
        
        return {
            'PPO': {
                'class': PPO,
                'policy': 'MlpPolicy',
                'params': {
                    'learning_rate': 3e-4,
                    'n_steps': 2048,
                    'batch_size': 256,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_range': 0.2,
                    'ent_coef': 0.01,
                    'vf_coef': 0.5,
                    'max_grad_norm': 0.5,
                    'policy_kwargs': policy_kwargs_large
                },
                'description': 'On-policy algorithm, stable, good for complex environments'
            },
            
            'SAC': {
                'class': SAC,
                'policy': 'MlpPolicy', 
                'params': {
                    'learning_rate': 3e-4,
                    'buffer_size': 100000,
                    'learning_starts': 10000,
                    'batch_size': 256,
                    'tau': 0.005,
                    'gamma': 0.99,
                    'train_freq': 1,
                    'gradient_steps': 1,
                    'ent_coef': 'auto',
                    'target_update_interval': 1,
                    'policy_kwargs': policy_kwargs_continuous
                },
                'description': 'Off-policy algorithm, sample efficient, good exploration'
            },
            
            'TD3': {
                'class': TD3,
                'policy': 'MlpPolicy',
                'params': {
                    'learning_rate': 3e-4,
                    'buffer_size': 100000,
                    'learning_starts': 10000,
                    'batch_size': 256,
                    'tau': 0.005,
                    'gamma': 0.99,
                    'train_freq': 1,
                    'gradient_steps': 1,
                    'policy_delay': 2,
                    'target_policy_noise': 0.2,
                    'target_noise_clip': 0.5,
                    'policy_kwargs': policy_kwargs_continuous,
                    'action_noise': NormalActionNoise(
                        mean=np.zeros(4), sigma=0.1 * np.ones(4)
                    )
                },
                'description': 'Off-policy algorithm, deterministic policy, stable training'
            }
        }
    
    def train_algorithm(self, algorithm_name: str, total_timesteps: int, 
                       use_wandb: bool = True) -> Dict[str, Any]:
        """
        Train a specific RL algorithm
        
        Args:
            algorithm_name: 'PPO', 'SAC', or 'TD3'
            total_timesteps: Training timesteps
            use_wandb: Whether to use WandB logging
        
        Returns:
            Training results dictionary
        """
        if algorithm_name not in self.algorithm_configs:
            raise ValueError(f"Algorithm {algorithm_name} not supported. Choose from: {list(self.algorithm_configs.keys())}")
        
        config = self.algorithm_configs[algorithm_name]
        print(f"\n[{algorithm_name}] Starting training for {total_timesteps:,} timesteps")
        print(f"[{algorithm_name}] {config['description']}")
        
        # Setup WandB for this algorithm
        if use_wandb:
            wandb.init(
                project="harfang-rl-algorithm-comparison",
                entity="BILGEM_DCS_RL",
                name=f"{algorithm_name}_harfang_{int(time.time())}",
                config={
                    **self.base_config,
                    **config['params'],
                    'algorithm': algorithm_name,
                    'total_timesteps': total_timesteps,
                    'state_space_dim': 25,
                    'action_space_dim': 4
                },
                tags=[algorithm_name.lower(), "harfang3d", "rl_llm", "algorithm_comparison"]
            )
        
        # Prepare environment (each algorithm gets its own VecNormalize)
        env = DummyVecEnv([lambda: self.env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        
        # Initialize algorithm
        model = config['class'](
            config['policy'],
            env,
            device=self.device,
            tensorboard_log=f"{self.log_dir}/{algorithm_name}",
            verbose=1,
            **config['params']
        )
        
        print(f"[{algorithm_name}] Model initialized with enhanced architecture")
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=f"{self.model_dir}/{algorithm_name}",
            name_prefix=f"{algorithm_name}_checkpoint"
        )
        callbacks.append(checkpoint_callback)
        
        # Algorithm-specific callback
        training_callback = MultiRLCallback(
            algorithm_name=algorithm_name,
            llm_assistant=self.llm_assistant,
            wandb_instance=wandb if use_wandb else None,
            model_dir=f"{self.model_dir}/{algorithm_name}"
        )
        callbacks.append(training_callback)
        
        # Train the model
        start_time = time.time()
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                tb_log_name=f"{algorithm_name}_run"
            )
            
            training_time = time.time() - start_time
            print(f"[{algorithm_name}] Training completed in {training_time:.1f}s")
            
            # Save final model
            final_model_path = f"{self.model_dir}/{algorithm_name}/{algorithm_name}_final.zip"
            model.save(final_model_path)
            
            # Save VecNormalize
            vec_normalize_path = f"{self.model_dir}/{algorithm_name}/{algorithm_name}_vecnormalize.pkl"
            env.save(vec_normalize_path)
            
            # Store trained model
            self.trained_models[algorithm_name] = {
                'model': model,
                'env': env,
                'model_path': final_model_path,
                'vec_normalize_path': vec_normalize_path,
                'training_time': training_time
            }
            
            print(f"[{algorithm_name}] Model saved to: {final_model_path}")
            
        except Exception as e:
            print(f"[ERROR] {algorithm_name} training failed: {e}")
            raise
        
        finally:
            if use_wandb:
                wandb.finish()
        
        return {
            'algorithm': algorithm_name,
            'total_timesteps': total_timesteps,
            'training_time': training_time,
            'final_model_path': final_model_path
        }
    
    def compare_algorithms(self, algorithms: list = None, total_timesteps: int = 500000):
        """
        Train and compare multiple RL algorithms
        
        Args:
            algorithms: List of algorithms to compare (default: all)
            total_timesteps: Training timesteps for each algorithm
        """
        if algorithms is None:
            algorithms = ['PPO', 'SAC', 'TD3']
        
        print(f"\n{'='*60}")
        print("MULTI-ALGORITHM RL COMPARISON FOR HARFANG COMBAT")
        print(f"{'='*60}")
        print(f"Algorithms: {algorithms}")
        print(f"Timesteps per algorithm: {total_timesteps:,}")
        print(f"Total training time estimated: {len(algorithms) * total_timesteps / 1000:.1f}k steps")
        
        training_results = []
        
        for algorithm in algorithms:
            try:
                result = self.train_algorithm(algorithm, total_timesteps, use_wandb=True)
                training_results.append(result)
                
                # Brief pause between algorithms
                time.sleep(5)
                
            except Exception as e:
                print(f"[ERROR] Failed to train {algorithm}: {e}")
                continue
        
        # Generate comparison report
        self._generate_comparison_report(training_results)
        
        return training_results
    
    def _generate_comparison_report(self, training_results: list):
        """Generate comprehensive algorithm comparison report"""
        print(f"\n{'='*60}")
        print("ALGORITHM COMPARISON RESULTS")
        print(f"{'='*60}")
        
        for result in training_results:
            algorithm = result['algorithm']
            training_time = result['training_time']
            
            print(f"{algorithm}:")
            print(f"  Training Time: {training_time:.1f}s ({training_time/60:.1f}m)")
            print(f"  Model Path: {result['final_model_path']}")
            
            # Load and evaluate each model for comparison
            if algorithm in self.trained_models:
                eval_results = self._quick_evaluation(algorithm)
                print(f"  Eval Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
                print(f"  Success Rate: {eval_results['success_rate']:.1%}")
        
        print(f"\n[COMPARISON] All algorithms trained and saved in: {self.model_dir}")
    
    def _quick_evaluation(self, algorithm_name: str, num_episodes: int = 10) -> Dict[str, float]:
        """Quick evaluation of trained algorithm"""
        if algorithm_name not in self.trained_models:
            return {'mean_reward': 0, 'std_reward': 0, 'success_rate': 0}
        
        model_data = self.trained_models[algorithm_name]
        model = model_data['model']
        env = model_data['env']
        
        episode_rewards = []
        successes = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            
            episode_rewards.append(episode_reward)
            # Success if episode reward > 50 (arbitrary threshold)
            successes.append(episode_reward > 50)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': np.mean(successes)
        }


class MultiRLCallback(BaseCallback):
    """Callback for multi-algorithm training with LLM integration tracking"""
    
    def __init__(self, algorithm_name: str, llm_assistant, wandb_instance=None, model_dir: str = "models"):
        super().__init__()
        self.algorithm_name = algorithm_name
        self.llm_assistant = llm_assistant
        self.wandb_instance = wandb_instance
        self.model_dir = model_dir
        
        # Algorithm-specific metrics
        self.episode_rewards = []
        self.llm_interaction_counts = []
        self.tactical_effectiveness_scores = []
        
    def _on_step(self) -> bool:
        """Called at each training step"""
        
        # Log algorithm-specific metrics
        if self.wandb_instance and self.num_timesteps % 1000 == 0:
            # Get current info
            infos = self.locals.get('infos', [{}])
            if infos and isinstance(infos[0], dict):
                info = infos[0]
                
                self.wandb_instance.log({
                    f'{self.algorithm_name}/timesteps': self.num_timesteps,
                    f'{self.algorithm_name}/learning_rate': getattr(self.model, 'learning_rate', 0),
                    
                    # Tactical metrics
                    f'{self.algorithm_name}/distance': info.get('distance', 0),
                    f'{self.algorithm_name}/threat_level': info.get('threat_level', 0),
                    f'{self.algorithm_name}/energy_state': 1 if info.get('energy_state') == 'HIGH' else 0,
                    f'{self.algorithm_name}/engagement_phase': 1 if info.get('engagement_phase') == 'BVR' else 0
                })
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        if len(self.model.ep_info_buffer) > 0:
            recent_episodes = self.model.ep_info_buffer[-5:]
            rewards = [ep_info['r'] for ep_info in recent_episodes]
            mean_reward = np.mean(rewards)
            
            print(f"[{self.algorithm_name}] Timesteps: {self.num_timesteps:,} | Mean Reward: {mean_reward:.2f}")
            
            if self.wandb_instance:
                self.wandb_instance.log({
                    f'{self.algorithm_name}/rollout_mean_reward': mean_reward,
                    f'{self.algorithm_name}/timesteps': self.num_timesteps
                })


def create_algorithm_configs() -> Dict[str, Dict[str, Any]]:
    """
    Create optimized configurations for different RL algorithms
    specifically tuned for Harfang air combat environment
    """
    return {
        'PPO_AGGRESSIVE': {
            'learning_rate': 5e-4,  # Higher LR for faster learning
            'n_steps': 4096,        # Larger rollout for better estimates
            'ent_coef': 0.02,       # Higher entropy for exploration
            'clip_range': 0.3       # Larger clip range for aggressive updates
        },
        
        'PPO_CONSERVATIVE': {
            'learning_rate': 1e-4,  # Lower LR for stable learning
            'n_steps': 1024,        # Smaller rollout for frequent updates
            'ent_coef': 0.005,      # Lower entropy for exploitation
            'clip_range': 0.1       # Smaller clip range for conservative updates
        },
        
        'SAC_EXPLORATION': {
            'learning_rate': 3e-4,
            'ent_coef': 0.2,        # High entropy for exploration
            'tau': 0.01,            # Faster target network updates
            'learning_starts': 5000  # Start learning earlier
        },
        
        'SAC_EXPLOITATION': {
            'learning_rate': 1e-4,
            'ent_coef': 0.05,       # Lower entropy for exploitation
            'tau': 0.005,           # Slower target network updates
            'learning_starts': 20000 # More experience before learning
        },
        
        'TD3_STABLE': {
            'learning_rate': 1e-4,
            'policy_delay': 3,      # More delayed policy updates
            'target_policy_noise': 0.1,  # Lower noise for stability
            'tau': 0.005           # Conservative target updates
        }
    }


if __name__ == "__main__":
    print("Multi-Algorithm RL Trainer for Harfang Combat Environment")
    print("Supports PPO, SAC, and TD3 with LLM integration")
    print("Usage: Import and use with HarfangEnhancedEnv and HarfangTacticalAssistant")
