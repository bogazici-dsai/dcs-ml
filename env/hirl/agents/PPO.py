from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

class PPOAgentSB3:
    def __init__(
        self,
        env,
        learning_rate,         # <-- unified name for LR
        n_steps,               # <-- unified name for rollout buffer
        gamma,
        summary_dir,
        model_name="ppo_agent",
        ent_coef=0.005,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_vecnormalize=True,
        norm_obs=True,
        norm_reward=True,
        **ppo_kwargs
    ):
        """
        PPO Agent using Stable-Baselines3 with VecNormalize support.
        All key hyperparameters are now passed in from the training script.
        """
        self.model_name = model_name
        self.summary_dir = summary_dir

        # --- Optionally wrap environment with VecNormalize ---
        if use_vecnormalize:
            if not isinstance(env, VecNormalize):
                env = DummyVecEnv([lambda: env])
                env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward, clip_obs=10.)
            self.env = env
            self.use_vecnormalize = True
        else:
            self.env = env
            self.use_vecnormalize = False

        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=min(64, n_steps // 4),
            n_epochs=10,
            gamma=gamma,
            ent_coef=ent_coef,
            clip_range=clip_range,
            gae_lambda=gae_lambda,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            tensorboard_log=summary_dir,
            verbose=1,
            device='auto',
            **ppo_kwargs
        )
        print(f"🤖 PPO Model created with:")
        print(f"   - Learning Rate: {learning_rate}")
        print(f"   - n_steps: {n_steps}")
        print(f"   - Entropy Coefficient: {ent_coef}")
        print(f"   - Clip Range: {clip_range}")
        print(f"   - Device: {self.model.device}")
        if self.use_vecnormalize:
            print(f"   - Using VecNormalize (obs: {norm_obs}, reward: {norm_reward})")

    def chooseAction(self, state):
        """Choose action during training (with noise/exploration)"""
        action, _ = self.model.predict(state, deterministic=False)
        return action

    def chooseActionNoNoise(self, state):
        """Choose deterministic action for evaluation"""
        action, _ = self.model.predict(state, deterministic=True)
        return action

    def learn(self, total_timesteps):
        """Train the PPO model"""
        print(f"🎯 Starting PPO training for {total_timesteps:,} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            reset_num_timesteps=True
        )
        print(f"✅ Training completed!")

    def saveCheckpoints(self, ajan, model_dir):
        """Save model checkpoint and (optionally) normalization statistics"""
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/{ajan}_{self.model_name}"
        self.model.save(model_path)
        print(f"💾 Model saved: {model_path}.zip")

        # Save VecNormalize statistics if using it
        if self.use_vecnormalize and isinstance(self.env, VecNormalize):
            self.env.save(os.path.join(model_dir, f"{ajan}_vecnormalize.pkl"))
            print(f"💾 VecNormalize stats saved: {os.path.join(model_dir, f'{ajan}_vecnormalize.pkl')}")

    def loadCheckpoints(self, ajan, model_dir):
        """Load model checkpoint and (optionally) normalization statistics"""
        model_path = f"{model_dir}/{ajan}_{self.model_name}.zip"
        vecnorm_path = os.path.join(model_dir, f"{ajan}_vecnormalize.pkl")

        if os.path.exists(model_path):
            # Load VecNormalize stats if available and needed
            if self.use_vecnormalize and os.path.exists(vecnorm_path):
                self.env = VecNormalize.load(vecnorm_path, self.env)
                print(f"📁 VecNormalize stats loaded: {vecnorm_path}")

            self.model = PPO.load(model_path, env=self.env)
            print(f"📁 Model loaded: {model_path}")
        else:
            print(f"⚠️ Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def get_model_info(self):
        """Get model information for debugging"""
        return {
            "learning_rate": self.model.learning_rate,
            "n_steps": self.model.n_steps,
            "batch_size": self.model.batch_size,
            "n_epochs": self.model.n_epochs,
            "ent_coef": self.model.ent_coef,
            "clip_range": self.model.clip_range,
            "device": str(self.model.device),
            "total_timesteps": getattr(self.model, 'num_timesteps', 0)
        }
