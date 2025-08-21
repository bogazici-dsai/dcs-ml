# Enhanced PPO Agent optimized for Harfang 25-dimensional state space with LLM integration
import os
import time
import numpy as np
import torch
from typing import Dict, Any, Optional, Callable

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ WandB not available. Install with: pip install wandb")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.utils.tensorboard import SummaryWriter


class EnhancedPPOAgent:
    """
    Enhanced PPO Agent specifically optimized for Harfang 25-dimensional tactical state space
    with integrated LLM guidance and comprehensive tactical reward shaping.
    """
    
    def __init__(
        self,
        env,
        llm_assistant,
        config: Dict[str, Any],
        model_name: str = "enhanced_ppo_harfang",
        use_wandb: bool = True,
        device: str = "auto"
    ):
        """
        Initialize Enhanced PPO Agent
        
        Args:
            env: HarfangEnhancedEnv with 25-dimensional state space
            llm_assistant: HarfangTacticalAssistant for guidance
            config: Training configuration dictionary
            model_name: Name for model saving and logging
            use_wandb: Whether to use Weights & Biases logging
            device: Device for training ('auto', 'cpu', 'cuda', 'mps')
        """
        self.env = env
        self.llm_assistant = llm_assistant
        self.config = config
        self.model_name = model_name
        self.use_wandb = use_wandb
        
        # Device selection
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"[ENHANCED PPO] Using device: {self.device}")
        
        # Setup directories
        self.log_dir = f"logs/{model_name}"
        self.model_dir = f"models/{model_name}"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Enhanced network architecture for 25D state space
        policy_kwargs = {
            "net_arch": {
                "pi": [512, 256, 128],  # Policy network - larger for complex tactical decisions
                "vf": [512, 256, 128]   # Value network - matches policy complexity
            },
            "activation_fn": torch.nn.ReLU,
            "ortho_init": True,         # Orthogonal initialization for stability
        }
        
        # Wrap environment with VecNormalize for stable training
        if not isinstance(env, VecNormalize):
            env = DummyVecEnv([lambda: env])
            env = VecNormalize(
                env, 
                norm_obs=True, 
                norm_reward=True, 
                clip_obs=10.0,      # Clip observations for stability
                clip_reward=10.0,   # Clip rewards for stability
                gamma=config.get('gamma', 0.99)
            )
        
        self.vec_env = env
        
        # Enhanced PPO model with optimized hyperparameters for air combat
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.get('learning_rate', 3e-4),
            n_steps=config.get('n_steps', 2048),
            batch_size=config.get('batch_size', 256),
            n_epochs=config.get('n_epochs', 10),
            gamma=config.get('gamma', 0.99),
            gae_lambda=config.get('gae_lambda', 0.95),
            clip_range=config.get('clip_range', 0.2),
            clip_range_vf=config.get('clip_range_vf', None),
            normalize_advantage=config.get('normalize_advantage', True),
            ent_coef=config.get('ent_coef', 0.01),  # Higher entropy for exploration
            vf_coef=config.get('vf_coef', 0.5),
            max_grad_norm=config.get('max_grad_norm', 0.5),
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.log_dir,
            device=self.device,
            verbose=1
        )
        
        print(f"[ENHANCED PPO] Model initialized with 25D state space optimization")
        print(f"[ENHANCED PPO] Policy architecture: {policy_kwargs['net_arch']}")
        
        # Training metrics tracking
        self.training_metrics = {
            'total_timesteps': 0,
            'episodes_completed': 0,
            'best_mean_reward': -np.inf,
            'llm_interventions': 0,
            'tactical_improvements': 0
        }
        
        # Setup WandB if requested
        if self.use_wandb and WANDB_AVAILABLE:
            self._setup_wandb()
        elif self.use_wandb and not WANDB_AVAILABLE:
            print("[WARNING] WandB requested but not available")
            self.use_wandb = False
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        try:
            wandb.init(
                project="harfang-rl-llm-enhanced",
                entity="BILGEM_DCS_RL",
                name=f"{self.model_name}_{int(time.time())}",
                config={
                    **self.config,
                    "model_name": self.model_name,
                    "device": self.device,
                    "state_space_dim": 25,
                    "action_space_dim": 4,
                    "llm_model": getattr(self.llm_assistant, 'model_name', 'unknown'),
                    "llm_rate_hz": self.llm_assistant.max_rate_hz
                },
                tags=["enhanced_ppo", "harfang3d", "rl_llm", "tactical_ai"]
            )
            print("[ENHANCED PPO] WandB logging initialized")
        except Exception as e:
            print(f"[WARNING] WandB setup failed: {e}")
            self.use_wandb = False
    
    def train(self, total_timesteps: int, save_freq: int = 50000, eval_freq: int = 25000):
        """
        Train the enhanced PPO agent with LLM guidance
        
        Args:
            total_timesteps: Total training timesteps
            save_freq: Frequency to save model checkpoints
            eval_freq: Frequency to run evaluation episodes
        """
        print(f"[ENHANCED PPO] Starting training for {total_timesteps:,} timesteps")
        print(f"[ENHANCED PPO] Save frequency: {save_freq:,} | Eval frequency: {eval_freq:,}")
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=self.model_dir,
            name_prefix=f"{self.model_name}_checkpoint"
        )
        callbacks.append(checkpoint_callback)
        
        # Enhanced training callback with LLM metrics
        training_callback = EnhancedTrainingCallback(
            llm_assistant=self.llm_assistant,
            eval_freq=eval_freq,
            wandb_instance=wandb if self.use_wandb else None,
            model_dir=self.model_dir,
            model_name=self.model_name
        )
        callbacks.append(training_callback)
        
        # Start training
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                tb_log_name="enhanced_ppo_run"
            )
            
            print(f"[ENHANCED PPO] Training completed successfully!")
            
            # Save final model
            final_model_path = os.path.join(self.model_dir, f"{self.model_name}_final.zip")
            self.model.save(final_model_path)
            
            # Save VecNormalize statistics
            vec_normalize_path = os.path.join(self.model_dir, f"{self.model_name}_vecnormalize.pkl")
            self.vec_env.save(vec_normalize_path)
            
            print(f"[ENHANCED PPO] Final model saved to: {final_model_path}")
            print(f"[ENHANCED PPO] VecNormalize stats saved to: {vec_normalize_path}")
            
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            raise
        
        finally:
            if self.use_wandb:
                wandb.finish()
    
    def evaluate(self, num_episodes: int = 10, deterministic: bool = True):
        """
        Evaluate the trained agent
        
        Args:
            num_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy
        """
        print(f"[ENHANCED PPO] Evaluating agent for {num_episodes} episodes")
        
        episode_rewards = []
        episode_lengths = []
        tactical_metrics = []
        
        for episode in range(num_episodes):
            obs = self.vec_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_tactical_data = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.vec_env.step(action)
                
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                episode_length += 1
                
                # Collect tactical metrics if available
                if isinstance(info, list) and len(info) > 0:
                    episode_tactical_data.append(info[0])
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            tactical_metrics.append(episode_tactical_data)
            
            print(f"  Episode {episode+1}: Reward={episode_reward:.2f}, Length={episode_length}")
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        print(f"[EVALUATION] Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"[EVALUATION] Mean Episode Length: {mean_length:.1f}")
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'tactical_metrics': tactical_metrics
        }
    
    def save_model(self, path: str):
        """Save the trained model and normalization statistics"""
        self.model.save(path)
        vec_normalize_path = path.replace('.zip', '_vecnormalize.pkl')
        self.vec_env.save(vec_normalize_path)
        print(f"[ENHANCED PPO] Model saved to: {path}")
        print(f"[ENHANCED PPO] VecNormalize saved to: {vec_normalize_path}")
    
    def load_model(self, path: str):
        """Load a trained model and normalization statistics"""
        self.model = PPO.load(path, env=self.vec_env, device=self.device)
        vec_normalize_path = path.replace('.zip', '_vecnormalize.pkl')
        if os.path.exists(vec_normalize_path):
            self.vec_env = VecNormalize.load(vec_normalize_path, self.vec_env)
            print(f"[ENHANCED PPO] VecNormalize loaded from: {vec_normalize_path}")
        print(f"[ENHANCED PPO] Model loaded from: {path}")


class EnhancedTrainingCallback(BaseCallback):
    """Enhanced callback with LLM integration and tactical metrics"""
    
    def __init__(self, llm_assistant, eval_freq: int, wandb_instance=None, 
                 model_dir: str = "models", model_name: str = "enhanced_ppo"):
        super().__init__()
        self.llm_assistant = llm_assistant
        self.eval_freq = eval_freq
        self.wandb_instance = wandb_instance
        self.model_dir = model_dir
        self.model_name = model_name
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.llm_interventions = []
        self.tactical_scores = []
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        """Called at each training step"""
        
        # Log training metrics to WandB
        if self.wandb_instance and self.num_timesteps % 1000 == 0:
            # Get current training info
            infos = self.locals.get('infos', [{}])
            if infos and isinstance(infos[0], dict):
                info = infos[0]
                
                # Log basic metrics
                self.wandb_instance.log({
                    'timesteps': self.num_timesteps,
                    'learning_rate': self.model.learning_rate,
                    'clip_range': self.model.clip_range,
                    'entropy_coef': self.model.ent_coef,
                    
                    # Tactical metrics if available
                    'distance': info.get('distance', 0),
                    'locked': info.get('locked', 0),
                    'threat_level': info.get('threat_level', 0),
                    'energy_state': info.get('energy_state', 'unknown'),
                    'engagement_phase': info.get('engagement_phase', 'unknown')
                })
        
        # Periodic evaluation
        if self.num_timesteps % self.eval_freq == 0:
            self._run_evaluation()
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        # Get episode statistics
        if len(self.model.ep_info_buffer) > 0:
            recent_episodes = self.model.ep_info_buffer[-10:]  # Last 10 episodes
            
            rewards = [ep_info['r'] for ep_info in recent_episodes]
            lengths = [ep_info['l'] for ep_info in recent_episodes]
            
            mean_reward = np.mean(rewards)
            mean_length = np.mean(lengths)
            
            # Update tracking
            self.episode_rewards.extend(rewards)
            self.episode_lengths.extend(lengths)
            
            # Log to console
            print(f"[TRAINING] Timesteps: {self.num_timesteps:,} | "
                  f"Mean Reward: {mean_reward:.2f} | Mean Length: {mean_length:.1f}")
            
            # Log to WandB
            if self.wandb_instance:
                self.wandb_instance.log({
                    'rollout/mean_reward': mean_reward,
                    'rollout/mean_length': mean_length,
                    'rollout/episodes_completed': len(self.episode_rewards),
                    'timesteps': self.num_timesteps
                })
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_model_path = os.path.join(self.model_dir, f"{self.model_name}_best.zip")
                self.model.save(best_model_path)
                print(f"[ENHANCED PPO] New best model saved! Reward: {mean_reward:.2f}")
    
    def _run_evaluation(self):
        """Run evaluation episodes with LLM guidance metrics"""
        print(f"[EVALUATION] Running evaluation at timestep {self.num_timesteps:,}")
        
        # Temporarily disable training mode
        self.model.policy.set_training_mode(False)
        
        eval_rewards = []
        llm_intervention_counts = []
        
        for episode in range(5):  # Quick evaluation
            obs = self.model.env.reset()
            done = False
            episode_reward = 0
            llm_interventions = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.model.env.step(action)
                
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                
                # Count LLM interventions (if tracked in info)
                if isinstance(info, list) and len(info) > 0:
                    if info[0].get('llm_intervention', False):
                        llm_interventions += 1
            
            eval_rewards.append(episode_reward)
            llm_intervention_counts.append(llm_interventions)
        
        # Re-enable training mode
        self.model.policy.set_training_mode(True)
        
        # Log evaluation results
        mean_eval_reward = np.mean(eval_rewards)
        mean_llm_interventions = np.mean(llm_intervention_counts)
        
        print(f"[EVALUATION] Mean Reward: {mean_eval_reward:.2f} | "
              f"LLM Interventions: {mean_llm_interventions:.1f}")
        
        if self.wandb_instance:
            self.wandb_instance.log({
                'eval/mean_reward': mean_eval_reward,
                'eval/std_reward': np.std(eval_rewards),
                'eval/mean_llm_interventions': mean_llm_interventions,
                'eval/timesteps': self.num_timesteps
            })


def create_enhanced_ppo_config(
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5
) -> Dict[str, Any]:
    """
    Create optimized PPO configuration for Harfang air combat training
    
    These hyperparameters are specifically tuned for:
    - 25-dimensional tactical state space
    - Continuous action space (pitch, roll, yaw, fire)
    - Long episodes (up to 2000 steps)
    - LLM-guided reward shaping
    """
    return {
        'learning_rate': learning_rate,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_range': clip_range,
        'ent_coef': ent_coef,
        'vf_coef': vf_coef,
        'max_grad_norm': max_grad_norm,
        'normalize_advantage': True,
        'clip_range_vf': None
    }


if __name__ == "__main__":
    # Example usage
    print("Enhanced PPO Agent for Harfang RL-LLM Combat Training")
    print("This module provides optimized PPO training for 25-dimensional tactical state space")
    print("Usage: Import and use with HarfangEnhancedEnv and HarfangTacticalAssistant")
