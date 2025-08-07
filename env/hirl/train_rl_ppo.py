from agents.PPO import PPOAgentSB3
from utils.seed import *
from environments.HarfangEnv_GYM_ppo import *
import environments.dogfight_client as df

import numpy as np
import time
import math
import datetime
import os
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter

# WandB imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ WandB not available. Install with: pip install wandb")

def setup_wandb(config, agent_name, env_type, model_name):
    if not WANDB_AVAILABLE:
        return None
    try:
        wandb.init(
            entity="BILGEM_DCS_RL",
            project="pure-rl-ucav-dogfight",
            name=f"{agent_name}_{env_type}_{model_name}",
            config=config,
            tags=[agent_name, env_type, "pure_rl", "harfang3d"]
        )
        print(f"✅ WandB initialized: {agent_name}_{env_type}_{model_name}")
        return wandb
    except Exception as e:
        print(f"⚠️ WandB setup failed: {e}")
        return None

def save_parameters_to_txt(log_dir, **kwargs):
    filename = os.path.join(log_dir, "log1.txt")
    with open(filename, 'w') as file:
        for key, value in kwargs.items():
            file.write(f"{key}={value}\n")

def create_ppo_callback(wandb_instance):
    if wandb_instance is None:
        return None
    from stable_baselines3.common.callbacks import BaseCallback
    import numpy as np

    class PPOWandbCallback(BaseCallback):
        def __init__(self, wandb_instance, verbose=0):
            super().__init__(verbose)
            self.wandb = wandb_instance
            self.episode_rewards = []
            self.episode_lengths = []
            self.episode_successes = []
            self.episode_fire_success = []
            self.episode_steps = []
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_fire_success = 0
            self.current_steps = 0

        def _on_step(self) -> bool:
            step_reward = self.locals.get('rewards', [0])[0]
            self.current_episode_reward += step_reward
            self.current_episode_length += 1
            self.current_steps += 1

            if self.locals.get('dones', [False])[0]:
                info = self.locals.get('infos', [{}])[0]
                success = int(info.get('success', 0))
                fire_success = int(info.get('fire_success', 0)) if 'fire_success' in info else 0

                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.episode_successes.append(success)
                self.episode_fire_success.append(fire_success)
                self.episode_steps.append(self.current_steps)

                cum_avg_success = np.mean(self.episode_successes) if self.episode_successes else 0.0
                rolling10 = np.mean(self.episode_successes[-10:]) if len(self.episode_successes) >= 10 else cum_avg_success
                rolling50 = np.mean(self.episode_successes[-50:]) if len(self.episode_successes) >= 50 else cum_avg_success
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else (np.mean(self.episode_rewards) if self.episode_rewards else 0.0)

                logs = {
                    "ppo_training/episode_reward": self.current_episode_reward,
                    "ppo_training/episode_length": self.current_episode_length,
                    "ppo_training/total_episodes": len(self.episode_rewards),
                    "ppo_training/current_total_timesteps": self.num_timesteps,
                    "ppo_training/cumulative_avg_success_rate": cum_avg_success,
                    "ppo_training/rolling10_avg_success_rate": rolling10,
                    "ppo_training/rolling50_avg_success_rate": rolling50,
                    "ppo_training/progress_avg_reward": avg_reward,
                    "ppo_training/episode_fire_success": fire_success,
                    "ppo_training/episode_steps": self.current_steps,
                }
                self.wandb.log(logs)
                self.current_episode_reward = 0
                self.current_episode_length = 0
                self.current_fire_success = 0
                self.current_steps = 0

            # Log progress and SB3 metrics every 1000 steps
            if self.num_timesteps % 1000 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else (np.mean(self.episode_rewards) if self.episode_rewards else 0.0)
                avg_success = np.mean(self.episode_successes[-10:]) if len(self.episode_successes) >= 10 else (np.mean(self.episode_successes) if self.episode_successes else 0.0)
                log_dict = {
                    "ppo_training/progress_total_timesteps": self.num_timesteps,
                    "ppo_training/progress_avg_reward": avg_reward,
                    "ppo_training/progress_avg_success_rate": avg_success,
                }
                # Log SB3 PPO metrics (KL, entropy, losses, etc.)
                try:
                    if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
                        sb3_metrics = self.model.logger.name_to_value
                        for key, value in sb3_metrics.items():
                            if key.startswith("train/") or key.startswith("time/"):
                                log_dict["ppo_sb3/" + key.replace("train/", "").replace("time/", "time/")] = value
                except Exception as e:
                    print(f"[WandB PPO Callback] Could not fetch SB3 metrics: {e}")

                self.wandb.log(log_dict)
            return True

    return PPOWandbCallback(wandb_instance)

def log_to_wandb(wandb_instance, episode, result_dict, phase="training"):
    if wandb_instance is None:
        return
    try:
        log_dict = {
            f"{phase}/episode": episode,
            f"{phase}/reward": result_dict.get('reward', 0),
            f"{phase}/min_distance": result_dict.get('min_distance', 0),
            f"{phase}/fire_attempts": result_dict.get('fire_attempts', 0),
            f"{phase}/successful_fires": result_dict.get('successful_fires', 0),
            f"{phase}/locks": result_dict.get('locks', 0),
            f"{phase}/success": int(result_dict.get('success', False)),
            f"{phase}/fire_success": int(result_dict.get('fire_success', False)),
            f"{phase}/done": int(result_dict.get('done', False)),
            f"{phase}/steps": result_dict.get('steps', 0),
            f"{phase}/fire_rate": result_dict.get('fire_rate', 0),
        }
        wandb_instance.log(log_dict)
    except Exception as e:
        print(f"⚠️ WandB logging failed: {e}")

def log_validation_rolling_metrics(wandb_instance, scores, trainsuccess, firesuccess, episode):
    if wandb_instance is None or len(scores) < 10:
        return
    try:
        windows = [10, 50]
        for window in windows:
            if len(scores) >= window:
                recent_scores = scores[-window:]
                recent_success = trainsuccess[-window:]
                avg_reward = np.mean(recent_scores) if len(recent_scores) > 0 else 0.0
                reward_std = np.std(recent_scores) if len(recent_scores) > 0 else 0.0
                success_rate = np.mean(recent_success) if len(recent_success) > 0 else 0.0
                rolling_metrics = {
                    f"validation_rolling_{window}/avg_reward": avg_reward,
                    f"validation_rolling_{window}/reward_std": reward_std,
                    f"validation_rolling_{window}/success_rate": success_rate,
                }
                wandb_instance.log(rolling_metrics)
    except Exception as e:
        print(f"⚠️ Rolling metrics logging failed: {e}")

def main(config):
    agent_name = config.agent
    model_name = config.model_name
    port = config.port
    load_model = config.load_model
    render = not (config.render)
    plot = config.plot
    env_type = config.env

    if config.random:
        print("Using random initialization")
    else:
        print("Using fixed initialization")
    if_random = config.random

    if config.seed is not None:
        set_seed(config.seed)
        print(f"Successfully set seed: {config.seed}")

    if not render:
        print('Rendering mode enabled')
    else:
        print('Renderless mode enabled')

    with open('local_config.yaml', 'r') as file:
        local_config = yaml.safe_load(file)
    if local_config['network']['ip'] == 'YOUR_IP_ADDRESS':
        raise ValueError("Please update the 'network.ip' field in local_config.yaml with your own IP address.")

    df.connect(local_config["network"]["ip"], port)
    start = time.time()
    df.disable_log()
    name = "Harfang_GYM"

    # ENVIRONMENT SETUP WITH MAX STEPS AND HYPERPARAMS
    if env_type == "straight_line":
        print("Environment: Harfang straight line")
        trainingEpisodes = 6000
        validationEpisodes = 250
        maxStep = 9000
        validationStep = 1500
        env = HarfangEnv(max_episode_steps=maxStep)
        ent_coef = 0.001
        n_steps = 512
        learning_rate = 1e-3
        clip_range = 0.2
    elif env_type == "serpentine":
        print("Environment: Harfang serpentine")
        trainingEpisodes = 1500
        validationEpisodes = 20
        maxStep = 1500
        validationStep = 1500
        env = HarfangSerpentineEnv(max_episode_steps=maxStep)
        ent_coef = 0.003
        n_steps = 1024
        learning_rate = 1e-3
        clip_range = 0.2
    elif env_type == "circular":
        print("Environment: Harfang circular")
        trainingEpisodes = 2000
        validationEpisodes = 20
        maxStep = 2000
        validationStep = 2000
        env = HarfangCircularEnv(max_episode_steps=maxStep)
        ent_coef = 0.005
        n_steps = 2048
        learning_rate = 5e-4
        clip_range = 0.15

    df.set_renderless_mode(render)
    df.set_client_update_mode(True)

    bufferSize = 10 ** 5
    gamma = 0.99
    criticLR = 1e-3
    tau = 0.005
    batchSize = 128
    stateDim = 13
    actionDim = 4

    start_time = datetime.datetime.now()
    log_dir = os.path.join(
        local_config["experiment"]["result_dir"],
        env_type, agent_name, (model_name or "PPO"),
        f"{start_time.year}_{start_time.month}_{start_time.day}_{start_time.hour}_{start_time.minute}"
    )
    model_dir = os.path.join(log_dir, 'model')
    summary_dir = os.path.join(log_dir, 'summary')
    plot_dir = os.path.join(log_dir, 'plot')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print('Agent: PPO (Stable-Baselines3)')
    agent = PPOAgentSB3(
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        summary_dir=summary_dir,
        model_name="PPO",
        ent_coef=ent_coef,
        clip_range=clip_range,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5
    )
    print(f"📊 PPO Hyperparameters:")
    print(f"   - Entropy Coefficient: {ent_coef}")
    print(f"   - n_steps: {n_steps}")
    print(f"   - Learning Rate: {learning_rate}")
    print(f"   - Clip Range: {clip_range}")

    # Load model if requested
    if load_model:
        print("📁 Loading pre-trained model...")
        agent.loadCheckpoints("FinalModel", model_dir)
        print("✅ Model loaded successfully")

    wandb_config = {
        'trainingEpisodes': trainingEpisodes,
        'maxStep': maxStep,
        'bufferSize': bufferSize,
        'batchSize': batchSize,
        'gamma': gamma,
        'tau': tau,
        'ent_coef': ent_coef,
        'n_steps': n_steps,
        'learning_rate': learning_rate,
        'clip_range': clip_range,
    }
    wandb_instance = setup_wandb(wandb_config, agent_name, env_type, model_name or "PPO")

    save_parameters_to_txt(
        log_dir=log_dir, bufferSize=bufferSize, criticLR=criticLR, batchSize=batchSize,
        maxStep=maxStep, validationStep=validationStep,
        agent=agent_name, model_dir=model_dir, env_type=env_type,
        ent_coef=ent_coef, n_steps=n_steps, learning_rate=learning_rate
    )
    env.save_parameters_to_txt(log_dir)
    writer = SummaryWriter(summary_dir)

    print(f"\n=== PPO TRAINING PHASE ===")
    total_timesteps = trainingEpisodes * maxStep
    print(f"🎯 Training for {total_timesteps:,} timesteps")
    print(f"   Max episodes: {trainingEpisodes}")
    print(f"   Max steps per episode: {maxStep}")

    # Log initial training setup
    if wandb_instance:
        wandb_instance.log({
            "setup/total_timesteps": total_timesteps,
            "setup/training_episodes": trainingEpisodes,
            "setup/max_step": maxStep,
            "setup/environment": env_type,
            "setup/start_time": time.time(),
            "setup/n_steps": n_steps,
            "setup/batch_size": getattr(agent.model, 'batch_size', 'unknown'),
            "setup/n_epochs": getattr(agent.model, 'n_epochs', 'unknown'),
        })

    ppo_callback = create_ppo_callback(wandb_instance)
    agent.model.learn(total_timesteps=total_timesteps, callback=ppo_callback, progress_bar=True)

    training_time = time.time() - start
    print(f"✅ PPO training completed in {training_time:.1f} seconds")

    # Log final PPO training metrics
    if wandb_instance:
        wandb_instance.log({
            "ppo/total_timesteps": agent.model.num_timesteps
        })
        wandb_instance.log({
            "training_summary/total_training_time": training_time,
            "training_summary/timesteps_per_second": total_timesteps / training_time,
            "training_summary/final_timesteps": getattr(agent.model, 'num_timesteps', total_timesteps),
            "training_summary/training_completed": True,
        })

    agent.saveCheckpoints("FinalModel", model_dir)
    print(f"💾 Final model saved to {model_dir}")

    print(f"\n=== VALIDATION PHASE ({validationEpisodes} episodes) ===")
    scores = []
    trainsuccess = []
    firesuccess = []

    for episode in range(validationEpisodes):
        episode_start_time = time.time()
        if if_random:
            state = env.random_reset()
        else:
            state = env.reset()
        totalReward = 0
        done = False
        episode_fire_attempts = 0
        episode_successful_fires = 0
        episode_locks = 0
        min_distance = float('inf')
        step = 0
        while not done and step < validationStep:
            action = agent.chooseActionNoNoise(state)
            n_state, reward, done, info = env.step(action)
            step_success = info.get("step_success", None)
            episode_success = int(info.get('success', 0)) if info else 0
            if action[3] > 0:
                episode_fire_attempts += 1
                if step_success == 1:
                    episode_successful_fires += 1
            if n_state[7] > 0:
                episode_locks += 1
            min_distance = min(min_distance, env.loc_diff)
            state = n_state
            totalReward += reward
            step += 1

        episode_time = time.time() - episode_start_time
        scores.append(totalReward)
        trainsuccess.append(int(env.episode_success))
        firesuccess.append(int(env.fire_success))

        fire_rate = episode_successful_fires / max(episode_fire_attempts, 1)
        training_result = {
            'reward': totalReward,
            'min_distance': min_distance,
            'fire_attempts': episode_fire_attempts,
            'successful_fires': episode_successful_fires,
            'locks': episode_locks,
            'success': env.episode_success,
            'fire_success': env.fire_success,
            'done': done,
            'steps': step,
            'fire_rate': fire_rate,
        }
        log_to_wandb(wandb_instance, episode, training_result, "validation")
        log_validation_rolling_metrics(wandb_instance, scores, trainsuccess, firesuccess, episode)

        if wandb_instance:
            avg_success = np.mean(trainsuccess) if len(trainsuccess) > 0 else 0.0
            wandb_instance.log({"validation/cumulative_avg_success_rate": avg_success, "validation/episode": episode})
            if len(trainsuccess) >= 10:
                rolling_avg = np.mean(trainsuccess[-10:]) if len(trainsuccess[-10:]) > 0 else 0.0
                wandb_instance.log({"validation/rolling10_avg_success_rate": rolling_avg, "validation/episode": episode})

        if episode % 5 == 0 or episode < 10:
            print(
                f"Val Episode {episode + 1}: Reward={totalReward:.1f}, MinDist={min_distance:.1f}, Fires={episode_fire_attempts}, Success={episode_successful_fires}, Rate={fire_rate:.3f}, Locks={episode_locks}, Steps={step}, Victory={env.episode_success}")

        writer.add_scalar('Validation/Episode Reward', totalReward, episode)
        writer.add_scalar('Validation/Episode Fire Attempts', episode_fire_attempts, episode)
        writer.add_scalar('Validation/Episode Successful Fires', episode_successful_fires, episode)
        writer.add_scalar('Validation/Episode Locks', episode_locks, episode)
        writer.add_scalar('Validation/Min Distance', min_distance, episode)

    avg_reward = np.mean(scores) if len(scores) > 0 else 0.0
    success_rate = np.mean(trainsuccess) if len(trainsuccess) > 0 else 0.0
    fire_success_rate = np.mean(firesuccess) if len(firesuccess) > 0 else 0.0
    print(f"\n🎉 FINAL RESULTS:")
    print(f"   Environment: {env_type}")
    print(f"   Training Time: {training_time:.1f} seconds")
    print(f"   Training Timesteps: {total_timesteps:,}")
    print(f"   Average Validation Reward: {avg_reward:.1f}")
    print(f"   Success Rate: {success_rate:.3f}")
    print(f"   Fire Success Rate: {fire_success_rate:.3f}")

    if wandb_instance:
        final_metrics = {
            "final/success_rate": success_rate,
            "final/avg_reward": avg_reward,
            "final/fire_success_rate": fire_success_rate,
            "final/total_episodes": validationEpisodes,
            "final/training_time": training_time,
            "final/total_timesteps": total_timesteps,
            "final/reward_std": np.std(scores) if len(scores) > 0 else 0.0,
            "final/reward_median": np.median(scores) if len(scores) > 0 else 0.0,
            "final/reward_max": np.max(scores) if len(scores) > 0 else 0.0,
            "final/reward_min": np.min(scores) if len(scores) > 0 else 0.0,
            "final/timesteps_per_second": total_timesteps / training_time if training_time > 0 else 0.0,
            "final/episodes_per_hour": (validationEpisodes / (training_time / 3600)) if training_time > 0 else 0.0,
        }
        if len(scores) > 1:
            final_metrics.update({
                "analysis/reward_improvement": scores[-1] - scores[0] if len(scores) > 1 else 0,
                "analysis/consistency_score": 1.0 - (np.std(scores) / max(abs(np.mean(scores)), 1.0)) if np.mean(scores) != 0 else 0,
                "analysis/learning_efficiency": success_rate / max(training_time / 3600, 0.1),
            })
        wandb_instance.log(final_metrics)
        wandb_instance.finish()
        print("✅ WandB run completed and finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='PPO', choices=['PPO'],
                        help="Specify the agent to use: 'PPO' (Stable-Baselines3 only).")
    parser.add_argument('--port', type=int, default=None,
                        help="The port number for the training environment. Example: 12345.")
    parser.add_argument('--model_name', type=str, default=None,
                        help="Name of the model to be saved. Example: 'PPO_experiment1'.")
    parser.add_argument('--load_model', action='store_true',
                        help="Flag to load a pre-trained model. Use this if you want to resume training.")
    parser.add_argument('--render', action='store_true',
                        help="Flag to enable rendering of the environment.")
    parser.add_argument('--plot', action='store_true',
                        help="Flag to plot training metrics. Use this for visualization.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducibility. Default is None (no seed).")
    parser.add_argument('--env', type=str, default='straight_line',
                        choices=['straight_line', 'serpentine', 'circular'],
                        help="Specify the training environment type. Default is 'straight_line'.")
    parser.add_argument('--random', action='store_true',
                        help="Flag to use random initialization in the environment. Default is False.")
    main(parser.parse_args())
