from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
import os


class PPOAgentSB3:
    def __init__(
            self,
            env,
            learning_rate,  # <-- unified name for LR
            n_steps,  # <-- unified name for rollout buffer
            gamma,
            summary_dir,
            model_name="ppo_agent",
            ent_coef=0.008,  # Updated default
            clip_range=0.2,
            gae_lambda=0.95,
            vf_coef=0.5,
            max_grad_norm=0.5,
            batch_size=None,  # Now explicitly passed
            n_epochs=4,  # Reduced default from 10 to 4
            use_vecnormalize=True,
            norm_obs=True,
            norm_reward=True,
            **ppo_kwargs
    ):
        """
        OPTIMIZED PPO Agent using Stable-Baselines3 with enhanced architecture.
        All key hyperparameters are now properly tuned for the dogfighting environment.
        """
        self.model_name = model_name
        self.summary_dir = summary_dir

        # --- Calculate proper batch size if not provided ---
        if batch_size is None:
            batch_size = max(128, n_steps // 4)  # Ensure minimum 128, proper ratio

        # Ensure batch size is reasonable
        batch_size = min(batch_size, n_steps)  # Can't be larger than rollout
        self.batch_size = batch_size

        # --- Enhanced Policy Network Architecture ---
        policy_kwargs = {
            "net_arch": {
                "pi": [256, 256, 128],  # Larger actor network for complex control
                "vf": [256, 256, 128]  # Larger critic network for better value estimation
            },
            "activation_fn": nn.ReLU,
            "ortho_init": True,  # Orthogonal initialization for stability
        }

        # Merge with any additional policy kwargs
        if "policy_kwargs" in ppo_kwargs:
            policy_kwargs.update(ppo_kwargs.pop("policy_kwargs"))

        # --- Optionally wrap environment with VecNormalize ---
        if use_vecnormalize:
            if not isinstance(env, VecNormalize):
                env = DummyVecEnv([lambda: env])
                env = VecNormalize(
                    env,
                    norm_obs=norm_obs,
                    norm_reward=norm_reward,
                    clip_obs=10.,
                    clip_reward=10.,  # Clip rewards to prevent extreme values
                    gamma=gamma  # Use same gamma for reward normalization
                )
            self.env = env
            self.use_vecnormalize = True
        else:
            self.env = env
            self.use_vecnormalize = False

        # --- Create PPO Model with Optimized Parameters ---
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            ent_coef=ent_coef,
            clip_range=clip_range,
            gae_lambda=gae_lambda,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            tensorboard_log=summary_dir,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device='auto',
            seed=None,  # Will be set by external seed setting
            **ppo_kwargs
        )

        # --- Print Configuration Summary ---
        print(f"🤖 OPTIMIZED PPO Model created with:")
        print(f"   - Learning Rate: {learning_rate}")
        print(f"   - n_steps (rollout): {n_steps}")
        print(f"   - Batch Size: {batch_size} (ratio: {batch_size / n_steps:.2f})")
        print(f"   - n_epochs: {n_epochs}")
        print(f"   - Entropy Coefficient: {ent_coef}")
        print(f"   - Clip Range: {clip_range}")
        print(f"   - GAE Lambda: {gae_lambda}")
        print(f"   - Value Function Coef: {vf_coef}")
        print(f"   - Max Grad Norm: {max_grad_norm}")
        print(f"   - Device: {self.model.device}")
        print(f"   - Network Architecture: {policy_kwargs['net_arch']}")
        if self.use_vecnormalize:
            print(f"   - VecNormalize: Enabled (obs: {norm_obs}, reward: {norm_reward})")

        # --- Store hyperparameters for logging ---
        self.hyperparameters = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'ent_coef': ent_coef,
            'clip_range': clip_range,
            'gae_lambda': gae_lambda,
            'vf_coef': vf_coef,
            'max_grad_norm': max_grad_norm,
            'use_vecnormalize': use_vecnormalize,
            'network_arch': str(policy_kwargs['net_arch'])
        }

    def chooseAction(self, state):
        """Choose action during training (with noise/exploration)"""
        action, _ = self.model.predict(state, deterministic=False)
        return action

    def chooseActionNoNoise(self, state):
        """Choose deterministic action for evaluation"""
        action, _ = self.model.predict(state, deterministic=True)
        return action

    def learn(self, total_timesteps, callback=None):
        """Train the PPO model with enhanced logging"""
        print(f"🎯 Starting OPTIMIZED PPO training for {total_timesteps:,} timesteps...")
        print(f"   - Expected updates: {total_timesteps // self.hyperparameters['n_steps']}")
        print(f"   - Batches per update: {self.hyperparameters['n_steps'] // self.hyperparameters['batch_size']}")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=True
        )
        print(f"✅ Optimized training completed!")

    def saveCheckpoints(self, ajan, model_dir):
        """Save model checkpoint and (optionally) normalization statistics"""
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/{ajan}_{self.model_name}"
        self.model.save(model_path)
        print(f"💾 Model saved: {model_path}.zip")

        # Save VecNormalize statistics if using it
        if self.use_vecnormalize and isinstance(self.env, VecNormalize):
            vecnorm_path = os.path.join(model_dir, f"{ajan}_vecnormalize.pkl")
            self.env.save(vecnorm_path)
            print(f"💾 VecNormalize stats saved: {vecnorm_path}")

        # Save hyperparameters for reference
        import json
        hyperparam_path = os.path.join(model_dir, f"{ajan}_hyperparameters.json")
        with open(hyperparam_path, 'w') as f:
            json.dump(self.hyperparameters, f, indent=2)
        print(f"💾 Hyperparameters saved: {hyperparam_path}")

    def loadCheckpoints(self, ajan, model_dir):
        """Load model checkpoint and (optionally) normalization statistics"""
        model_path = f"{model_dir}/{ajan}_{self.model_name}.zip"
        vecnorm_path = os.path.join(model_dir, f"{ajan}_vecnormalize.pkl")
        hyperparam_path = os.path.join(model_dir, f"{ajan}_hyperparameters.json")

        if os.path.exists(model_path):
            # Load VecNormalize stats if available and needed
            if self.use_vecnormalize and os.path.exists(vecnorm_path):
                self.env = VecNormalize.load(vecnorm_path, self.env)
                print(f"📁 VecNormalize stats loaded: {vecnorm_path}")

            self.model = PPO.load(model_path, env=self.env)
            print(f"📁 Model loaded: {model_path}")

            # Load and display hyperparameters if available
            if os.path.exists(hyperparam_path):
                import json
                with open(hyperparam_path, 'r') as f:
                    loaded_hyperparams = json.load(f)
                print(f"📁 Loaded hyperparameters: {loaded_hyperparams}")
        else:
            print(f"⚠️ Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def get_model_info(self):
        """Get comprehensive model information for debugging"""
        info = {
            "learning_rate": self.model.learning_rate,
            "n_steps": self.model.n_steps,
            "batch_size": self.model.batch_size,
            "n_epochs": self.model.n_epochs,
            "ent_coef": self.model.ent_coef,
            "clip_range": self.model.clip_range,
            "gae_lambda": self.model.gae_lambda,
            "vf_coef": self.model.vf_coef,
            "max_grad_norm": self.model.max_grad_norm,
            "device": str(self.model.device),
            "total_timesteps": getattr(self.model, 'num_timesteps', 0),
            "policy_class": str(type(self.model.policy)),
            "use_vecnormalize": self.use_vecnormalize
        }

        # Add network architecture info if available
        if hasattr(self.model.policy, 'mlp_extractor'):
            try:
                info["actor_network"] = str(self.model.policy.mlp_extractor.policy_net)
                info["critic_network"] = str(self.model.policy.mlp_extractor.value_net)
            except:
                info["network_info"] = "Architecture details not available"

        return info

    def get_training_diagnostics(self):
        """Get training diagnostics for monitoring"""
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            return self.model.logger.name_to_value.copy()
        return {}

    def set_learning_rate(self, new_lr):
        """Dynamically adjust learning rate during training"""
        self.model.learning_rate = new_lr
        # Update the optimizer learning rate
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"📊 Learning rate updated to: {new_lr}")

    def set_entropy_coef(self, new_ent_coef):
        """Dynamically adjust entropy coefficient during training"""
        self.model.ent_coef = new_ent_coef
        print(f"📊 Entropy coefficient updated to: {new_ent_coef}")

    def get_policy_entropy(self):
        """Get current policy entropy for monitoring exploration"""
        if hasattr(self.model, '_last_obs') and self.model._last_obs is not None:
            try:
                with torch.no_grad():
                    distribution = self.model.policy.get_distribution(
                        self.model.policy.obs_to_tensor(self.model._last_obs)[0]
                    )
                    return distribution.entropy().mean().item()
            except:
                return None
        return None

    def validate_hyperparameters(self):
        """Validate that hyperparameters are within reasonable ranges"""
        warnings = []

        # Check batch size
        if self.model.batch_size < 64:
            warnings.append(f"Batch size ({self.model.batch_size}) is quite small, may cause unstable training")

        if self.model.batch_size > self.model.n_steps:
            warnings.append(f"Batch size ({self.model.batch_size}) is larger than n_steps ({self.model.n_steps})")

        # Check learning rate
        if self.model.learning_rate > 1e-2:
            warnings.append(f"Learning rate ({self.model.learning_rate}) is quite high for continuous control")

        # Check entropy coefficient
        if self.model.ent_coef < 0.001:
            warnings.append(f"Entropy coefficient ({self.model.ent_coef}) is very low, may limit exploration")

        # Check epochs
        if self.model.n_epochs > 10:
            warnings.append(f"n_epochs ({self.model.n_epochs}) is high, may cause overfitting")

        if warnings:
            print("⚠️ Hyperparameter warnings:")
            for warning in warnings:
                print(f"   - {warning}")
        else:
            print("✅ Hyperparameters look reasonable")

        return warnings