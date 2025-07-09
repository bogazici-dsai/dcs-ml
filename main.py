import os
import torch
import numpy as np
from loguru import logger
from typing import Dict
import re
import wandb
import random

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from stable_baselines3 import PPO

from MinigridAssistant.minigrid_assistant_mediator import MiniGridAgentWithMediator
from utils.make_minigrid_env import make_env


def get_device():
    """Device detection"""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.backends.cuda.is_built() and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class Config:
    """Clean config"""

    def __init__(self):
        # Environment
        self.env_name = "MiniGrid-DoorKey-6x6-v0"
        self.max_steps = 100

        # Training
        self.max_episodes = 5
        self.exploration_episodes = 3

        # LLM
        self.llm_model = "llama3.1:8b"
        self.llm_temperature = 0.1

        # Logging
        self.log_every = 1
        self.save_every = 25

        # WandB
        self.use_wandb = True
        self.wandb_entity = "BILGEM_DCS_RL"
        self.wandb_project = "mediator-fixed-rewards"


def save_mediator_model(minigrid_agent, episode, config, experiment_type="training"):
    """Save the trained mediator model"""
    try:
        if experiment_type == "best":
            filename = f"mediator_best.pt"
        elif experiment_type == "final":
            filename = f"mediator_final_{episode}ep.pt"
        else:
            filename = f"mediator_training_ep{episode}.pt"

        filepath = os.path.join("checkpoints", filename)

        torch.save({
            'asking_policy_state_dict': minigrid_agent.mediator.asking_policy.state_dict(),
            'episode': episode,
            'config': {
                'env_name': config.env_name,
                'max_episodes': config.max_episodes,
                'exploration_episodes': config.exploration_episodes
            },
            'experiment_type': experiment_type
        }, filepath)

        logger.info(f"💾 Model saved: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        return None


def load_mediator_model(minigrid_agent, filepath):
    """Load a previously trained mediator model"""
    try:
        checkpoint = torch.load(filepath, map_location=str(minigrid_agent.mediator.device))
        minigrid_agent.mediator.asking_policy.load_state_dict(checkpoint['asking_policy_state_dict'])
        logger.info(f"✅ Model loaded from: {filepath}")
        return checkpoint
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None


def setup_wandb(config: Config, experiment_type: str = "mediator"):
    """Setup WandB"""
    if not config.use_wandb:
        return None

    try:
        project_name = f"{config.wandb_project}-{experiment_type}"
        wandb.init(
            entity=config.wandb_entity,
            project=project_name,
            config={
                "episodes": config.max_episodes,
                "env": config.env_name,
                "llm": config.llm_model,
                "system": f"fixed_rewards_v1_{experiment_type}",
                "experiment_type": experiment_type
            }
        )
        print(f"✅ WandB ready for {experiment_type}")
        return wandb
    except Exception as e:
        print(f"⚠️ WandB failed: {e}")
        config.use_wandb = False
        return None


def calculate_aligned_mediator_reward(env_reward: float, was_interrupted: bool,
                                      plan_changed: bool, step: int, episode: int,
                                      recent_performance: float = 0.5, config: Config = None) -> float:
    """SMART: Dynamic mediator reward calculation"""

    # 1. BASE REWARD: Scaled environment reward
    base_reward = env_reward * 0.8

    # 2. ASKING BEHAVIOR ANALYSIS
    asking_modifier = 0.0

    if was_interrupted:
        if plan_changed:
            if env_reward > 0:
                asking_modifier = 0.3 + (env_reward * 0.2)
            elif env_reward == 0:
                asking_modifier = 0.1
            else:
                asking_modifier = -0.1 + (env_reward * 0.1)
        else:
            if env_reward >= 0:
                asking_modifier = -0.15 - (0.05 * step / 50)
            else:
                asking_modifier = -0.08
    else:
        if env_reward > 0:
            efficiency_bonus = 0.1 + (env_reward * 0.05)
            episode_scale = min(1.5, 1.0 + (episode / 100))
            asking_modifier = efficiency_bonus * episode_scale
        elif env_reward < -0.05:
            should_have_asked_penalty = abs(env_reward) * 0.3
            performance_scale = 2.0 - recent_performance
            asking_modifier = -should_have_asked_penalty * performance_scale
        else:
            asking_modifier = 0.02

    # 3. STEP EFFICIENCY
    step_efficiency = 0.0
    if step > 60:
        step_efficiency = -0.002 * (step - 60) ** 1.2
    elif step < 30 and env_reward > 0:
        step_efficiency = 0.05 * (30 - step) / 30

    # 4. EPISODE PROGRESS
    learning_adjustment = 0.0
    if config:
        exploration_episodes = config.exploration_episodes
        if episode < exploration_episodes * 0.5:
            if asking_modifier < 0:
                learning_adjustment = abs(asking_modifier) * 0.4
            if was_interrupted:
                learning_adjustment += 0.05
        elif episode < exploration_episodes * 0.9:
            if asking_modifier < 0:
                learning_adjustment = abs(asking_modifier) * 0.2
        elif episode < exploration_episodes + 500:
            if asking_modifier < 0:
                learning_adjustment = abs(asking_modifier) * 0.1
            if asking_modifier > 0 and not was_interrupted:
                learning_adjustment = asking_modifier * 0.2
        else:
            if asking_modifier > 0 and not was_interrupted:
                learning_adjustment = asking_modifier * 0.3
            elif was_interrupted and not plan_changed:
                learning_adjustment = -0.05

    # 5. COMBINE ALL FACTORS
    final_reward = base_reward + asking_modifier + step_efficiency + learning_adjustment

    # 6. REASONABLE BOUNDS
    final_reward = max(-1.0, min(2.0, final_reward))

    return final_reward


def run_ppo_only_episode(rl_agent, rl_env, episode: int, max_steps: int = 100):
    """PPO-only baseline episode"""
    obs, _ = rl_env.reset(seed=episode)

    done = False
    step = 0
    total_reward = 0

    while not done and step < max_steps:
        step += 1

        if rl_agent:
            action, _ = rl_agent.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = rl_env.action_space.sample()

        try:
            obs, reward, terminated, truncated, _ = rl_env.step(action)
        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            reward = -0.1
            obs, _, terminated, truncated, _ = rl_env.step(0)

        done = terminated or truncated
        total_reward += reward

    episode_success = total_reward > 0

    return {
        'env_reward': total_reward,
        'steps': step,
        'success': episode_success,
        'interrupts': 0,
        'phase': "PPO_ONLY"
    }


def run_episode_flow(rl_agent, minigrid_agent, rl_env, llm_env, episode: int, max_steps: int = 100,
                     results: list = None,
                     config: Config = None):
    """Main episode flow for mediator training"""

    if results is None:
        results = []

    # Reset environments
    rl_obs, _ = rl_env.reset(seed=episode)
    llm_obs, llm_info = llm_env.reset(seed=episode)

    done = False
    sim_step = 0
    total_env_reward = 0
    total_mediator_reward = 0
    interrupts = 0
    overrides = 0

    llm_info["llm_env"] = llm_env

    # Phase tracking
    if config:
        exploration_episodes = config.exploration_episodes
        minigrid_agent.mediator.exploration_episodes = config.exploration_episodes
        minigrid_agent.mediator.current_episode = episode

        if episode < config.exploration_episodes:
            progress = episode / config.exploration_episodes
            minigrid_agent.mediator.epsilon = max(0.15, 0.6 * (1 - progress * 0.75))
        else:
            minigrid_agent.mediator.epsilon = 0.15

    phase = "EXPLORATION" if episode < exploration_episodes else "EXPLOITATION"

    while not done and sim_step < max_steps:
        sim_step += 1

        # RL AGENT DECISION
        if rl_agent:
            rl_action, _ = rl_agent.predict(rl_obs, deterministic=True)
            rl_action = int(rl_action)
        else:
            rl_action = rl_env.action_space.sample()

        # MINIGRID AGENT DECISION
        final_action, was_interrupted, interaction_info = minigrid_agent.agent_run(
            sim_step=sim_step,
            obs=llm_obs,
            rl_action=rl_action,
            infos=llm_info,
            reward=total_env_reward,
            use_learned_asking=True
        )

        # STATS
        if was_interrupted:
            interrupts += 1
            if interaction_info.get('llm_plan_changed', False):
                overrides += 1

        # ENVIRONMENT STEP
        try:
            rl_obs, env_reward, terminated, truncated, _ = rl_env.step(final_action)
            llm_obs, _, _, _, llm_info = llm_env.step(final_action)
            llm_info["llm_env"] = llm_env
        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            env_reward = -0.1
            rl_obs, _, terminated, truncated, _ = rl_env.step(0)
            llm_obs, _, _, _, llm_info = llm_env.step(0)
            llm_info["llm_env"] = llm_env

        done = terminated or truncated
        total_env_reward += env_reward

        # MEDIATOR TRAINING
        if sim_step > 1:
            recent_performance = 0.5
            if len(results) >= 5:
                recent_performance = np.mean([r['success'] for r in results[-5:]])

            mediator_reward = calculate_aligned_mediator_reward(
                env_reward=env_reward,
                was_interrupted=was_interrupted,
                plan_changed=interaction_info.get('llm_plan_changed', False),
                step=sim_step,
                episode=episode,
                recent_performance=recent_performance,
                config=config
            )

            total_mediator_reward += mediator_reward

            minigrid_agent.mediator.train_asking_policy(
                obs=llm_obs,
                action=rl_action,
                reward=mediator_reward,
                next_obs=llm_obs,
                asked_llm=was_interrupted,
                llm_plan_changed=interaction_info.get('llm_plan_changed', False)
            )

    # Episode end
    episode_success = total_env_reward > 0
    minigrid_agent.update_performance(episode_success)
    minigrid_agent.mediator.episode_end(episode_success)

    efficiency = overrides / max(interrupts, 1) if interrupts > 0 else 1.0

    return {
        'env_reward': total_env_reward,
        'mediator_reward': total_mediator_reward,
        'steps': sim_step,
        'success': episode_success,
        'interrupts': interrupts,
        'overrides': overrides,
        'efficiency': efficiency,
        'phase': phase
    }


def evaluate_loaded_model(config, rl_agent, minigrid_agent, rl_env, llm_env, num_episodes: int = 5):
    """Evaluate loaded model with random seeds"""
    print(f"🔍 Evaluating loaded model on {num_episodes} random episodes...")

    # Generate random seeds
    eval_seeds = [random.randint(0, 999999) for _ in range(num_episodes)]

    # Save original settings
    original_wandb = config.use_wandb
    config.use_wandb = False

    # EVALUATION MODE - NO TRAINING
    original_epsilon = minigrid_agent.mediator.epsilon
    original_train_flag = minigrid_agent.train_mediator
    original_current_episode = minigrid_agent.mediator.current_episode

    minigrid_agent.mediator.epsilon = 0.05  # Minimal exploration
    minigrid_agent.train_mediator = False  # NO TRAINING!
    minigrid_agent.mediator.current_episode = 9999  # Force exploitation

    eval_results = []

    for episode in range(num_episodes):
        try:
            # Reset with random seed
            eval_seed = eval_seeds[episode]
            rl_obs, _ = rl_env.reset(seed=eval_seed)
            llm_obs, llm_info = llm_env.reset(seed=eval_seed)
            llm_info["llm_env"] = llm_env

            # Episode start log
            print(f"\n🔍 EVAL Episode {episode} START (Seed: {eval_seed})")

            done = False
            sim_step = 0
            total_env_reward = 0
            interrupts = 0
            overrides = 0

            while not done and sim_step < 100:
                sim_step += 1

                # RL decision
                if rl_agent:
                    rl_action, _ = rl_agent.predict(rl_obs, deterministic=True)
                    rl_action = int(rl_action)
                else:
                    rl_action = rl_env.action_space.sample()

                # Mediator decision (NO TRAINING!)
                final_action, was_interrupted, interaction_info = minigrid_agent.agent_run(
                    sim_step=sim_step,
                    obs=llm_obs,
                    rl_action=rl_action,
                    infos=llm_info,
                    reward=total_env_reward,
                    use_learned_asking=True
                )

                # Stats
                if was_interrupted:
                    interrupts += 1
                    if interaction_info.get('llm_plan_changed', False):
                        overrides += 1

                # Environment step
                try:
                    rl_obs, env_reward, terminated, truncated, _ = rl_env.step(final_action)
                    llm_obs, _, _, _, llm_info = llm_env.step(final_action)
                    llm_info["llm_env"] = llm_env
                except Exception as e:
                    logger.error(f"Environment step failed: {e}")
                    env_reward = -0.1
                    rl_obs, _, terminated, truncated, _ = rl_env.step(0)
                    llm_obs, _, _, _, llm_info = llm_env.step(0)
                    llm_info["llm_env"] = llm_env

                done = terminated or truncated
                total_env_reward += env_reward

                # ❌ NO TRAINING!

            episode_success = total_env_reward > 0
            efficiency = overrides / max(interrupts, 1) if interrupts > 0 else 1.0

            # Episode end log
            print(f"📊 EVAL Episode {episode} END (Seed: {eval_seed}): "
                  f"Reward={total_env_reward:.2f}, "
                  f"Steps={sim_step}, "
                  f"Success={episode_success}, "
                  f"Interrupts={interrupts}, "
                  f"Efficiency={efficiency:.1%}")

            eval_results.append({
                'success': episode_success,
                'env_reward': total_env_reward,
                'steps': sim_step,
                'interrupts': interrupts,
                'efficiency': efficiency,
                'seed': eval_seed
            })

            # Progress every 3 episodes for small numbers
            if (episode + 1) % 3 == 0:
                current_success = np.mean([r['success'] for r in eval_results])
                current_interrupts = np.mean([r['interrupts'] for r in eval_results])
                print(f"📈 Progress: {episode + 1}/{num_episodes} | "
                      f"Success: {current_success:.1%} | "
                      f"Avg LLM Calls: {current_interrupts:.1f}")

        except Exception as e:
            logger.error(f"Evaluation episode {episode} failed: {e}")
            continue

    # Restore settings
    minigrid_agent.mediator.epsilon = original_epsilon
    minigrid_agent.train_mediator = original_train_flag
    minigrid_agent.mediator.current_episode = original_current_episode
    config.use_wandb = original_wandb

    # Results
    if eval_results:
        success_rate = np.mean([r['success'] for r in eval_results])
        avg_reward = np.mean([r['env_reward'] for r in eval_results])
        avg_interrupts = np.mean([r['interrupts'] for r in eval_results])
        avg_efficiency = np.mean([r['efficiency'] for r in eval_results])

        print(f"\n📊 EVALUATION RESULTS:")
        print(f"Episodes: {len(eval_results)}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Avg Reward: {avg_reward:.3f}")
        print(f"Avg LLM Calls: {avg_interrupts:.1f}")
        print(f"Avg Efficiency: {avg_efficiency:.1%}")

        return eval_results
    else:
        print("❌ No evaluation results!")
        return None


def log_to_wandb(episode: int, result: Dict, config: Config):
    """Enhanced WandB logging"""
    if not config.use_wandb:
        return

    try:
        wandb.log({
            "episode": episode,
            "env_reward": result['env_reward'],
            "mediator_reward": result['mediator_reward'],
            "success": int(result['success']),
            "efficiency": result['efficiency'],
            "interrupts": result['interrupts'],
            "overrides": result['overrides'],
            "phase": result['phase']
        })
    except Exception as e:
        pass


def check_available_envs():
    """Check available environments"""
    import gymnasium as gym
    try:
        env_candidates = ["MiniGrid-DoorKey-6x6-v0"]
        for env_name in env_candidates:
            try:
                env = gym.make(env_name)
                env.close()
                print(f"✅ {env_name} available")
                return env_name
            except:
                print(f"❌ {env_name} not found")
        return None
    except:
        return None


def main():
    """Main function with simple menu"""

    print("🔧 MEDIATOR TRAINING SYSTEM")
    print("=" * 50)

    # Setup
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Check requirements
    try:
        import gymnasium as gym
        import minigrid
        print("✅ Gymnasium & MiniGrid ready")
    except ImportError:
        print("❌ Missing packages")
        return

    # Environment detection
    available_env = check_available_envs()
    if not available_env:
        print("❌ No MiniGrid environment found!")
        return

    config = Config()
    config.env_name = available_env
    print(f"🎯 Environment: {config.env_name}")

    # Device
    device = get_device()
    print(f"🚀 Device: {device}")

    # Initialize LLM
    try:
        chat = ChatOllama(
            model=config.llm_model,
            temperature=config.llm_temperature,
            num_predict=200
        )
        llm = RunnableLambda(lambda x: chat.invoke(x))
        print(f"✅ LLM initialized: {config.llm_model}")

        # Test connection
        test_response = llm.invoke("Hello, respond with just 'OK'")
        print(f"✅ LLM test successful")
    except Exception as e:
        print(f"❌ LLM failed: {e}")
        return

    # Initialize environments
    try:
        rl_env, llm_env = make_env(env_name=config.env_name, max_steps=config.max_steps)
        print(f"✅ Environments created")
        obs_shape = llm_env.observation_space['image'].shape
        print(f"📐 Obs shape: {obs_shape}")
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return

    # Load RL Agent
    rl_agent = None
    rl_paths = ["models/ppo_minigrid_doorkey_6x6.zip"]

    for path in rl_paths:
        try:
            rl_agent = PPO.load(path, device=device)
            print(f"✅ RL agent loaded: {path}")
            break
        except:
            continue

    if not rl_agent:
        print("⚠️ RL agent not found, using random actions")

    # Initialize Mediator Agent
    minigrid_agent = MiniGridAgentWithMediator(
        llm=llm,
        obs_shape=obs_shape,
        device=str(device),
        verbose=True,
        train_mediator=True
    )
    print("✅ Mediator Agent initialized")

    # Check for existing models
    existing_models = []
    if os.path.exists("checkpoints"):
        existing_models = [f for f in os.listdir("checkpoints") if f.endswith('.pt')]

    if existing_models:
        print(f"💾 Found {len(existing_models)} existing model(s)")

    # MAIN LOOP - Simple menu
    while True:
        print("\n📋 MENU:")
        print("1. Training")
        print("2. Load & Evaluate Model")
        print("3. PPO Baseline")
        print("4. Exit")

        choice = input("\nChoice (1-4): ").strip()

        if choice == "1":
            print("🚀 Training starting...")

            # Setup WandB
            wandb_instance = setup_wandb(config, "training")

            # Training loop
            results = []
            best_success_rate = 0.0

            for episode in range(config.max_episodes):
                try:
                    result = run_episode_flow(
                        rl_agent=rl_agent,
                        minigrid_agent=minigrid_agent,
                        rl_env=rl_env,
                        llm_env=llm_env,
                        episode=episode,
                        results=results,
                        config=config
                    )
                    results.append(result)
                    log_to_wandb(episode, result, config)

                    # Progress log
                    if episode % config.log_every == 0:
                        recent_results = results[-10:] if len(results) >= 10 else results
                        avg_env_reward = np.mean([r['env_reward'] for r in recent_results])
                        success_rate = np.mean([r['success'] for r in recent_results])
                        print(f"Episode {episode:3d} | Env: {avg_env_reward:5.2f} | Success: {success_rate:.1%}")

                    # Save model periodically
                    if episode % config.save_every == 0 and episode > 0:
                        save_mediator_model(minigrid_agent, episode, config, "training")

                    # Save best model
                    if len(results) >= 3:  # Changed from 10 to 3 for small episode counts
                        recent_success_rate = np.mean([r['success'] for r in results[-3:]])
                        if recent_success_rate > best_success_rate:
                            best_success_rate = recent_success_rate
                            save_mediator_model(minigrid_agent, episode, config, "best")

                except Exception as e:
                    logger.error(f"Episode {episode} failed: {e}")
                    continue

            # Save final model
            save_mediator_model(minigrid_agent, config.max_episodes, config, "final")

            if config.use_wandb and wandb_instance:
                wandb_instance.finish()

            print("✅ Training completed!")
            input("\nPress Enter to continue...")

        elif choice == "2":
            print("📂 Available models:")
            if not os.path.exists("checkpoints"):
                print("❌ No checkpoints directory found!")
                input("Press Enter to continue...")
                continue

            checkpoint_files = [f for f in os.listdir("checkpoints") if f.endswith('.pt')]

            if not checkpoint_files:
                print("❌ No trained models found!")
                input("Press Enter to continue...")
                continue

            for i, file in enumerate(checkpoint_files):
                print(f"{i + 1}. {file}")

            try:
                choice_idx = int(input("Select model (number): ")) - 1
                if choice_idx < 0 or choice_idx >= len(checkpoint_files):
                    print("❌ Invalid selection!")
                    input("Press Enter to continue...")
                    continue

                selected_file = checkpoint_files[choice_idx]
                filepath = os.path.join("checkpoints", selected_file)

                # Load model
                checkpoint = load_mediator_model(minigrid_agent, filepath)
                if checkpoint:
                    print(f"✅ Model loaded from episode {checkpoint['episode']}")

                    # Evaluate with random seeds
                    eval_results = evaluate_loaded_model(
                        config, rl_agent, minigrid_agent, rl_env, llm_env,
                        num_episodes=5  # Use 5 episodes for quick test
                    )

                    if eval_results:
                        print("✅ Evaluation completed!")
                    else:
                        print("❌ Evaluation failed!")

            except (ValueError, IndexError):
                print("❌ Invalid selection!")

            input("Press Enter to continue...")

        elif choice == "3":
            print("🤖 PPO Baseline (no LLM)...")

            results = []
            for episode in range(5):
                result = run_ppo_only_episode(rl_agent, rl_env, episode)
                results.append(result)

                # Show progress for each episode
                current_success = np.mean([r['success'] for r in results])
                print(f"Episode {episode + 1}/5 | Success: {current_success:.1%}")

            # Results
            success_rate = np.mean([r['success'] for r in results])
            avg_reward = np.mean([r['env_reward'] for r in results])
            avg_steps = np.mean([r['steps'] for r in results])

            print(f"\n📊 PPO BASELINE RESULTS:")
            print(f"Success Rate: {success_rate:.1%}")
            print(f"Avg Reward: {avg_reward:.3f}")
            print(f"Avg Steps: {avg_steps:.1f}")
            print(f"LLM Calls: 0")

            input("Press Enter to continue...")

        elif choice == "4":
            print("👋 Exit")
            break
        else:
            print("❌ Invalid choice")
            input("Press Enter to try again...")

    # Cleanup
    rl_env.close()
    llm_env.close()
    print("✅ System closed!")


if __name__ == "__main__":
    main()