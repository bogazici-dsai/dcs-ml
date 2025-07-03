"""
Updated Main Training - Simple & Clean
- Mevcut feature extraction ve environment'ları AYNI
- Sadece mediator training'i basitleştirdik
- Karmaşık config'ler yok, temiz seçenekler
"""

import os
import torch
import numpy as np
from loguru import logger
from typing import Dict
import re

# Mevcut sistem importları - AYNI
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from stable_baselines3 import PPO

# Mevcut dosyalardan importlar - AYNI
from TSCAssistant.tsc_assistant_mediator import TSCAgentWithMediator
from utils.make_tsc_env import make_env
from TSCAssistant.feature_translator import translate_features_for_llm
from TSCAssistant.tsc_agent_prompt import render_prompt

# Yeni basit mediator
from simple_mediator_training import EnhancedMediator  # Yukarıdaki kodu buraya koy


def get_device():
    """Mevcut device detection - AYNI"""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.backends.cuda.is_built() and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class SimpleConfig:
    """Basit config - karmaşık değil"""

    def __init__(self):
        # Environment
        self.env_name = "MiniGrid-DoorKey-6x6-v0"
        self.max_steps = 100

        # Training - Basit
        self.max_episodes = 100
        self.exploration_episodes = 50  # İlk 50 exploration
        self.exploitation_episodes = 50  # Son 50 exploitation

        # LLM
        self.llm_model = "llama3.1:8b"
        self.llm_temperature = 0.1

        # Logging
        self.log_every = 10
        self.save_every = 25


def run_episode_with_simple_mediator(rl_agent, tsc_agent, rl_env, llm_env, episode: int, max_steps: int = 100):
    """
    Basit episode run - karmaşık logic yok
    """
    # Reset - AYNI
    rl_obs, _ = rl_env.reset(seed=episode)
    llm_obs, llm_info = llm_env.reset(seed=episode)

    done = False
    sim_step = 0
    total_reward = 0

    llm_info["llm_env"] = llm_env

    logger.info(f"Episode {episode} START")

    while not done and sim_step < max_steps:
        sim_step += 1

        # RL action - AYNI
        if rl_agent:
            rl_action, _ = rl_agent.predict(rl_obs, deterministic=True)
            rl_action = int(rl_action)
        else:
            rl_action = rl_env.action_space.sample()

        # TSC Agent decision - AYNI
        final_action, was_interrupted, interaction_info = tsc_agent.agent_run(
            sim_step=sim_step,
            obs=llm_obs,
            rl_action=rl_action,
            infos=llm_info,
            reward=total_reward,
            use_learned_asking=True
        )

        # Environment step - AYNI
        try:
            rl_obs, reward, terminated, truncated, _ = rl_env.step(final_action)
            llm_obs, _, _, _, llm_info = llm_env.step(final_action)
            llm_info["llm_env"] = llm_env
        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            rl_obs, reward, terminated, truncated, _ = rl_env.step(0)
            llm_obs, _, _, _, llm_info = llm_env.step(0)
            llm_info["llm_env"] = llm_env

        done = terminated or truncated
        total_reward += reward

        # Basit mediator training
        if sim_step > 1:
            tsc_agent.mediator.train_asking_policy(
                obs=llm_obs,
                action=rl_action,
                reward=reward,
                next_obs=llm_obs,
                asked_llm=was_interrupted,
                llm_plan_changed=interaction_info.get('llm_plan_changed', False)
            )

    # Episode end
    episode_success = total_reward > 0
    tsc_agent.mediator.episode_end(episode_success)

    logger.info(f"Episode {episode} END: Reward={total_reward:.2f}, Success={episode_success}")

    return {
        'reward': total_reward,
        'steps': sim_step,
        'success': episode_success,
        'interrupts': interaction_info.get('interrupts', 0),
        'overrides': interaction_info.get('overrides', 0)
    }


def main():
    """Ana fonksiyon - basit seçenekler"""

    print("🤖 SIMPLE MEDIATOR TRAINING")
    print("✅ Mevcut sistem - sadece mediator training basitleştirildi")
    print("✅ Exploration-Exploitation balance")
    print("✅ Karmaşık config'ler yok")
    print("=" * 50)

    # Setup
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    config = SimpleConfig()
    device = get_device()
    print(f"🚀 Device: {device}")

    # LLM - AYNI
    try:
        chat = ChatOllama(model=config.llm_model, temperature=config.llm_temperature)
        llm = RunnableLambda(lambda x: chat.invoke(x))
        print(f"✅ LLM: {config.llm_model}")
    except Exception as e:
        print(f"❌ LLM failed: {e}")
        return

    # Environments - AYNI
    try:
        rl_env, llm_env = make_env(env_name=config.env_name, max_steps=config.max_steps)
        obs_shape = llm_env.observation_space['image'].shape
        print(f"✅ Environment: {config.env_name}")
    except Exception as e:
        print(f"❌ Environment failed: {e}")
        return

    # RL Agent - AYNI
    rl_agent = None
    try:
        rl_agent = PPO.load("models/ppo_minigrid_doorkey_6x6_250000_steps", device=device)
        print(f"✅ RL agent loaded")
    except:
        print("⚠️ RL agent not found, using random")

    # TSC Agent with simple mediator
    tsc_agent = TSCAgentWithMediator(
        llm=llm,
        obs_shape=obs_shape,
        device=str(device),
        verbose=True,
        train_mediator=True
    )

    # Replace mediator with simple version
    tsc_agent.mediator = EnhancedMediator(obs_shape, device=str(device), verbose=True)
    print("✅ TSC Agent with Simple Mediator")

    # Training options - BASIT
    print("\n📋 TRAINING OPTIONS:")
    print("1. Quick Test (50 episodes)")
    print("2. Full Training (100 episodes)")
    print("3. Custom Episodes")
    print("4. Exit")

    choice = input("\nChoice (1-4): ").strip()

    if choice == "1":
        config.max_episodes = 50
        config.exploration_episodes = 25
        print("🚀 Quick test starting...")
    elif choice == "2":
        print("🚀 Full training starting...")
    elif choice == "3":
        try:
            episodes = int(input("Episodes: "))
            config.max_episodes = episodes
            config.exploration_episodes = episodes // 2
            print(f"🚀 Custom training: {episodes} episodes")
        except:
            print("❌ Invalid input")
            return
    elif choice == "4":
        print("👋 Exit")
        return
    else:
        print("❌ Invalid choice")
        return

    # TRAINING LOOP - BASIT
    print(f"\n🚀 Training: {config.max_episodes} episodes")
    print(f"📊 Exploration: {config.exploration_episodes} episodes")
    print(f"📊 Exploitation: {config.max_episodes - config.exploration_episodes} episodes")

    results = []

    for episode in range(config.max_episodes):
        try:
            result = run_episode_with_simple_mediator(
                rl_agent=rl_agent,
                tsc_agent=tsc_agent,
                rl_env=rl_env,
                llm_env=llm_env,
                episode=episode,
                max_steps=config.max_steps
            )
            results.append(result)

            # Progress log
            if episode % config.log_every == 0:
                stats = tsc_agent.mediator.get_statistics()
                recent_results = results[-10:] if len(results) >= 10 else results
                avg_reward = np.mean([r['reward'] for r in recent_results])
                success_rate = np.mean([r['success'] for r in recent_results])

                print(f"Episode {episode:3d} [{stats['phase'].upper():12s}] | "
                      f"Reward: {avg_reward:5.2f} | "
                      f"Success: {success_rate:.1%} | "
                      f"Ask Rate: {stats['ask_rate']:.2f} | "
                      f"Efficiency: {stats['asking_efficiency']:.2f}")

            # Save checkpoint
            if episode % config.save_every == 0 and episode > 0:
                save_path = f"checkpoints/simple_mediator_ep_{episode}.pt"
                tsc_agent.mediator.save_asking_policy(save_path)
                print(f"💾 Saved: {save_path}")

        except Exception as e:
            logger.error(f"Episode {episode} failed: {e}")
            continue

    # RESULTS - BASIT
    print("\n" + "=" * 40)
    print("TRAINING RESULTS")
    print("=" * 40)

    if results:
        success_rate = np.mean([r['success'] for r in results])
        avg_reward = np.mean([r['reward'] for r in results])
        final_stats = tsc_agent.mediator.get_statistics()

        print(f"Episodes: {len(results)}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average Reward: {avg_reward:.3f}")
        print(f"Final Ask Rate: {final_stats['ask_rate']:.2f}")
        print(f"Asking Efficiency: {final_stats['asking_efficiency']:.2f}")

        # Phase comparison
        exploration_results = results[:config.exploration_episodes]
        exploitation_results = results[config.exploration_episodes:]

        if exploration_results and exploitation_results:
            exp_success = np.mean([r['success'] for r in exploration_results])
            expl_success = np.mean([r['success'] for r in exploitation_results])

            print(f"\n📊 PHASE COMPARISON:")
            print(f"Exploration Success: {exp_success:.1%}")
            print(f"Exploitation Success: {expl_success:.1%}")

            if expl_success > exp_success:
                print("✅ GOOD: Exploitation better than exploration!")
            else:
                print("⚠️ Need more training")

        # Final save
        final_save_path = "checkpoints/simple_mediator_final.pt"
        tsc_agent.mediator.save_asking_policy(final_save_path)
        print(f"\n💾 Final model saved: {final_save_path}")

    # Cleanup
    rl_env.close()
    llm_env.close()
    print("\n✨ Training completed!")


if __name__ == "__main__":
    main()