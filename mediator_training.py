import os
import torch
import numpy as np
import wandb
from loguru import logger
from typing import Dict

# COMMENTED: ChatGPT support
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI

# ADDED: Llama support
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from TSCAssistant.tsc_assistant_mediator import TSCAgentWithMediator
from utils.make_tsc_env import make_env
from stable_baselines3 import PPO


# COMMENTED: Load .env file
# load_dotenv("/Users/tyerdogan/llm_udemy/llm_engineering/.env")


def get_device():
    """Get the best available device."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.backends.cuda.is_built() and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def run_episode_flow(rl_agent, tsc_agent, rl_env, llm_env, episode: int, max_steps: int = 100):
    """
    COMPLETE FIXED: Better monitoring of interrupt efficiency
    """
    # Reset environments
    rl_obs, _ = rl_env.reset(seed=episode)
    llm_obs, llm_info = llm_env.reset(seed=episode)

    done = False
    sim_step = 0
    total_reward = 0
    interrupts = 0
    overrides = 0
    agreements = 0  # NEW: Track agreements (ask but no change)
    step_rewards = []

    episode_success = False
    steps_without_progress = 0
    last_reward = 0

    llm_info["llm_env"] = llm_env
    logger.info(f"Episode {episode} START - RL is primary agent")

    while not done and sim_step < max_steps:
        sim_step += 1

        # RL AGENT MAKES PRIMARY DECISION
        rl_action, _ = rl_agent.predict(rl_obs, deterministic=True)
        rl_action = int(rl_action)

        # MEDIATOR DECIDES
        final_action, was_interrupted, interaction_info = tsc_agent.agent_run(
            sim_step=sim_step,
            obs=llm_obs,
            rl_action=rl_action,
            infos={"env": "MiniGrid-DoorKey-6x6-v0", "llm_env": llm_env},
            reward=total_reward,
            use_learned_asking=True
        )

        # UPDATED: Track efficiency metrics
        if was_interrupted:
            interrupts += 1
            if interaction_info.get('llm_plan_changed', False):
                overrides += 1
            else:
                agreements += 1  # Asked but agreed

        # EXECUTE ACTION IN ENVIRONMENT
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
        step_rewards.append(reward)

        if reward > last_reward:
            steps_without_progress = 0
        else:
            steps_without_progress += 1
        last_reward = total_reward

        if total_reward > 0:
            episode_success = True

        # UPDATED: More sophisticated mediator reward with efficiency penalty
        if sim_step > 1:
            mediator_reward = reward

            # EFFICIENCY PENALTY: Penalize recent agreements heavily
            if agreements > 0:
                agreement_penalty = (agreements / max(interrupts, 1)) * 0.2
                mediator_reward -= agreement_penalty

            # SUCCESS BONUS
            if episode_success and sim_step < max_steps * 0.8:
                mediator_reward += 0.1
            elif steps_without_progress > 10:
                mediator_reward -= 0.05

            # EXPLORATION BONUS (early training only)
            if len(tsc_agent.mediator.ask_history) < 300:
                if was_interrupted and interaction_info.get('llm_plan_changed', False):
                    mediator_reward += 0.05  # Bonus for good interrupts

            tsc_agent.mediator.train_asking_policy(
                obs=llm_obs,
                action=rl_action,
                reward=mediator_reward,
                next_obs=llm_obs,
                asked_llm=was_interrupted,
                llm_plan_changed=interaction_info.get('llm_plan_changed', False)
            )

    # Update performance tracking
    tsc_agent.update_performance(episode_success)

    # UPDATED: Calculate efficiency metrics
    interrupt_rate = interrupts / sim_step if sim_step > 0 else 0
    override_rate = overrides / max(interrupts, 1)
    agreement_rate = agreements / max(interrupts, 1)
    efficiency = overrides / max(interrupts, 1) if interrupts > 0 else 1.0  # Efficiency = override rate

    success_emoji = "✅" if total_reward > 0 else "❌"
    efficiency_emoji = "🚀" if efficiency > 0.5 else "⚠️" if efficiency > 0.2 else "❌"

    logger.info(f"Episode {episode} END: Reward={total_reward:.2f}, "
                f"Interrupts={interrupts}, Overrides={overrides}, Agreements={agreements}, "
                f"Efficiency={efficiency:.1%} {efficiency_emoji}, Success={success_emoji}")

    return {
        'reward': total_reward,
        'steps': sim_step,
        'success': total_reward > 0,
        'interrupts': interrupts,
        'overrides': overrides,
        'agreements': agreements,  # NEW
        'interrupt_rate': interrupt_rate,
        'override_rate': override_rate,
        'agreement_rate': agreement_rate,  # NEW
        'efficiency': efficiency,  # NEW: Override rate as efficiency
        'step_rewards': step_rewards,
        'avg_step_reward': np.mean(step_rewards) if step_rewards else 0,
        'episode_success': episode_success,
        'reward_per_step': total_reward / max(sim_step, 1)
    }


def evaluate_baseline(rl_agent, rl_env, num_episodes: int = 10):
    """
    Evaluate baseline RL performance without LLM
    """
    logger.info("Evaluating baseline RL performance (no LLM)...")
    baseline_results = []

    for episode in range(num_episodes):
        obs, _ = rl_env.reset(seed=episode + 1000)
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:
            action, _ = rl_agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = rl_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        baseline_results.append({
            'reward': total_reward,
            'success': total_reward > 0,
            'steps': steps
        })

    baseline_success = np.mean([r['success'] for r in baseline_results])
    baseline_reward = np.mean([r['reward'] for r in baseline_results])
    logger.info(f"Baseline RL: Success={baseline_success:.1%}, Avg Reward={baseline_reward:.3f}")
    return baseline_success, baseline_reward


def main():
    """COMPLETE FIXED: Main experiment with efficiency monitoring and early stopping."""

    # FIXED: Enhanced WandB config
    wandb.init(
        project="TSC_Mediator_Training",
        entity="BILGEM_DCS_RL",
        config={
            "env_name": "MiniGrid-DoorKey-6x6-v0",
            "mediator_episodes": 50,  # Reduced for efficiency testing
            "max_steps": 100,
            "algo": "TSC_Mediator_Efficiency_Fixed",
            "llm_model": "llama3.1:8b",  # CHANGED: Llama model
            "mediator_lr": 1e-4,
            "baseline_episodes": 10,
            "mediator_hidden_dim": 64,
            "lambda_penalty_start": 0.02,  # Higher start
            "lambda_penalty_end": 0.2,  # Higher end
            "agreement_penalty": 0.1,  # NEW
            "exploration_episodes": 300,  # Reduced
            "gradient_clip": 0.2,
            "entropy_bonus": 0.02,
            "l2_regularization": 0.001,
            "efficiency_threshold": 0.3,  # NEW
            "early_stopping_patience": 15,  # Reduced
        }
    )

    logger.info("Starting COMPLETE FIXED Efficiency-Focused Mediator Experiment")
    logger.info("Focus: Minimize unnecessary interrupts while maintaining performance")

    # Setup
    device = get_device()
    logger.info(f"Using device: {device}")

    # CHANGED: Initialize Llama instead of ChatGPT
    # chat = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # LLAMA 3.1-8B: Initialize LLM
    try:
        chat = ChatOllama(
            model="llama3.1:8b",
            temperature=0.1,  # Low temperature for consistency
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            num_predict=150,  # Limit response length
        )
        logger.info("🦙 Llama 3.1-8B model initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Llama model: {e}")
        logger.error("Make sure Ollama is running and llama3.1:8b model is downloaded:")
        logger.error("  1. Start Ollama: 'ollama serve'")
        logger.error("  2. Download model: 'ollama pull llama3.1:8b'")
        return

    # COMMENTED: Test API key
    # api_key = os.getenv("OPENAI_API_KEY")
    # if api_key:
    #     logger.info(f"🔑 API Key loaded: {api_key[:10]}...")
    # else:
    #     logger.error("❌ OPENAI_API_KEY not found!")

    llm = RunnableLambda(lambda x: chat.invoke(x))

    # Test Llama connection
    try:
        test_response = llm.invoke("Hello, respond with just 'OK'")
        logger.info(f"🦙 Llama test successful: {test_response}")
    except Exception as e:
        logger.error(f"❌ Llama connection failed: {e}")
        return

    # Initialize environments
    env_name = "MiniGrid-DoorKey-6x6-v0"
    rl_env, llm_env = make_env(env_name=env_name, max_steps=100)

    # Load RL Agent (PRIMARY)
    model_path = "models/ppo_minigrid_doorkey_6x6_250000_steps"

    try:
        rl_agent = PPO.load(model_path, device=device)
        logger.info("RL agent loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load RL agent: {e}")
        logger.info("Training environment will use random actions for demonstration")
        rl_agent = None

    # FIXED: Initialize Enhanced TSC Agent
    obs_shape = llm_env.observation_space['image'].shape
    tsc_agent = TSCAgentWithMediator(
        llm=llm,
        obs_shape=obs_shape,
        device=device,
        verbose=True,
        train_mediator=True
    )

    logger.info("Setup complete - EFFICIENCY-FOCUSED mediator with aggressive agreement penalty")

    # Baseline evaluation
    baseline_success = None
    baseline_reward = None
    if rl_agent is not None:
        baseline_success, baseline_reward = evaluate_baseline(rl_agent, rl_env, num_episodes=10)

        # Log baseline to WandB
        wandb.log({
            "baseline/success_rate": baseline_success,
            "baseline/avg_reward": baseline_reward,
        })

    # FIXED: Training with efficiency focus
    num_episodes = 10  # Reduced for testing
    logger.info(f"Training efficiency-focused mediator for {num_episodes} episodes...")

    results = []
    best_performance = -float('inf')
    patience = 15
    episodes_without_improvement = 0

    for episode in range(num_episodes):
        try:
            result = run_episode_flow(
                rl_agent=rl_agent,
                tsc_agent=tsc_agent,
                rl_env=rl_env,
                llm_env=llm_env,
                episode=episode
            )
            results.append(result)

            # UPDATED: Enhanced performance tracking with efficiency
            recent_performance = np.mean([r['reward'] for r in results[-10:]])
            recent_efficiency = np.mean([r['efficiency'] for r in results[-10:]])

            # Save best model based on BOTH performance AND efficiency
            combined_metric = recent_performance * recent_efficiency  # Reward * Efficiency
            if combined_metric > best_performance:
                best_performance = combined_metric
                episodes_without_improvement = 0
                os.makedirs("models", exist_ok=True)
                tsc_agent.save_mediator(f"models/best_efficient_mediator_ep_{episode}.pt")
                logger.info(
                    f"🎯 New best model saved! Performance={recent_performance:.3f}, Efficiency={recent_efficiency:.1%}")
            else:
                episodes_without_improvement += 1

            # EFFICIENCY-BASED EARLY STOPPING
            if episodes_without_improvement > patience and episode > 20:
                if recent_efficiency < 0.2:
                    logger.warning(f"Early stopping due to low efficiency: {recent_efficiency:.1%}")
                    break

            mediator_stats = tsc_agent.get_mediator_stats()

            # UPDATED: Enhanced logging with efficiency metrics
            log_data = {
                "episode": episode,
                "training/success": int(result['success']),
                "training/reward": result['reward'],
                "training/steps": result['steps'],
                "training/interrupts": result['interrupts'],
                "training/overrides": result['overrides'],
                "training/agreements": result['agreements'],
                "training/interrupt_rate": result['interrupt_rate'],
                "training/override_rate": result['override_rate'],
                "training/agreement_rate": result['agreement_rate'],
                "training/efficiency": result['efficiency'],
                "training/avg_step_reward": result['avg_step_reward'],
                "training/combined_metric": combined_metric,

                # Enhanced mediator metrics
                "mediator/ask_rate": mediator_stats.get('recent_ask_rate', 0),
                "mediator/avg_reward": mediator_stats.get('recent_avg_reward', 0),
                "mediator/recent_loss": mediator_stats.get('recent_loss', 0),
                "mediator/lambda_penalty": mediator_stats.get('lambda_penalty', 0.02),
                "mediator/agreement_penalty": mediator_stats.get('agreement_penalty', 0.1),
                "mediator/interrupt_efficiency": mediator_stats.get('interrupt_efficiency', 0),
                "mediator/recent_interrupt_rate": mediator_stats.get('recent_interrupt_rate', 0),
                "mediator/recent_agreement_rate": mediator_stats.get('recent_agreement_rate', 0),
                "mediator/training_phase": 1 if mediator_stats.get('training_phase') == 'exploitation' else 0,
                "mediator/baseline_reward": mediator_stats.get('baseline_reward', 0),
                "mediator/episodes_without_improvement": episodes_without_improvement,
            }

            wandb.log(log_data)

            # UPDATED: Better progress logging every 5 episodes
            if episode % 5 == 0 and episode > 0:
                recent_results = results[-5:]
                avg_success = np.mean([r['success'] for r in recent_results])
                avg_efficiency = np.mean([r['efficiency'] for r in recent_results])
                avg_interrupt_rate = np.mean([r['interrupt_rate'] for r in recent_results])
                avg_agreement_rate = np.mean([r['agreement_rate'] for r in recent_results])
                avg_reward = np.mean([r['reward'] for r in recent_results])

                logger.info(f"Episode {episode}: Success={avg_success:.1%}, "
                            f"Task_Reward={avg_reward:.3f}, "  # ADDED: Task reward
                            f"Efficiency={avg_efficiency:.1%}, "
                            f"Interrupt={avg_interrupt_rate:.1%}, "
                            f"Agreement={avg_agreement_rate:.1%}")

                # EFFICIENCY WARNING
                if avg_efficiency < 0.3:
                    logger.warning(f"⚠️  LOW EFFICIENCY WARNING: {avg_efficiency:.1%} - "
                                   f"Mediator asking too often without changing plans!")
                elif avg_efficiency > 0.7:
                    logger.info(f"🚀 EXCELLENT EFFICIENCY: {avg_efficiency:.1%}")

                # ADDED: Log both rewards clearly
                mediator_reward = mediator_stats.get('recent_avg_reward', 0)
                logger.info(f"Mediator: Ask_Rate={mediator_stats.get('recent_ask_rate', 0):.3f}, "
                            f"Mediator_Reward={mediator_reward:.3f}, "  # CLARIFIED: This is mediator's internal reward
                            f"λ={mediator_stats.get('lambda_penalty', 0.02):.3f}, "
                            f"Agreement_Penalty={mediator_stats.get('agreement_penalty', 0.1):.3f}")

                # ADDED: Explain the reward difference if there's confusion
                if episode == 5:
                    logger.info(f"📊 REWARD EXPLANATION:")
                    logger.info(f"   Task_Reward={avg_reward:.3f} = Environment reward (success/failure)")
                    logger.info(
                        f"   Mediator_Reward={mediator_reward:.3f} = Internal reward (with agreement penalties)")
                    logger.info(f"   Mediator learns from internal reward, task success from environment reward")

        except Exception as e:
            logger.error(f"Episode {episode} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print final results
    print("\n" + "=" * 70)
    print("EFFICIENCY-FOCUSED MEDIATOR TRAINING RESULTS")
    print("=" * 70)

    if results:
        success_rate = np.mean([r['success'] for r in results])
        avg_reward = np.mean([r['reward'] for r in results])
        avg_efficiency = np.mean([r['efficiency'] for r in results])
        avg_interrupts = np.mean([r['interrupts'] for r in results])
        avg_interrupt_rate = np.mean([r['interrupt_rate'] for r in results])
        avg_override_rate = np.mean([r['override_rate'] for r in results])
        avg_agreement_rate = np.mean([r['agreement_rate'] for r in results])

        print(f"Enhanced TSC Agent Performance:")
        print(f"Success Rate:           {success_rate:.1%}")
        print(f"Average Task Reward:    {avg_reward:.3f}")  # CLARIFIED: Task reward
        print(f"Average Efficiency:     {avg_efficiency:.1%}")
        print(f"Average Interrupts:     {avg_interrupts:.1f} per episode")
        print(f"Average Interrupt Rate: {avg_interrupt_rate:.1%}")
        print(f"Average Override Rate:  {avg_override_rate:.1%}")
        print(f"Average Agreement Rate: {avg_agreement_rate:.1%} (WASTE)")

        # EFFICIENCY ANALYSIS
        if avg_efficiency > 0.5:
            print(f"🚀 EXCELLENT EFFICIENCY: {avg_efficiency:.1%}")
        elif avg_efficiency > 0.3:
            print(f"✅ GOOD EFFICIENCY: {avg_efficiency:.1%}")
        else:
            print(f"⚠️  LOW EFFICIENCY: {avg_efficiency:.1%} - needs improvement")

        # Final metrics
        final_metrics = {
            "final/success_rate": success_rate,
            "final/avg_reward": avg_reward,
            "final/avg_efficiency": avg_efficiency,
            "final/avg_interrupts": avg_interrupts,
            "final/avg_interrupt_rate": avg_interrupt_rate,
            "final/avg_override_rate": avg_override_rate,
            "final/avg_agreement_rate": avg_agreement_rate,
            "final/episodes_without_improvement": episodes_without_improvement,
            "final/best_combined_metric": best_performance,
        }

        if baseline_success is not None:
            improvement = success_rate - baseline_success
            reward_improvement = avg_reward - baseline_reward if baseline_reward else 0
            print(f"\nComparison to Baseline RL:")
            print(f"Baseline Success Rate:  {baseline_success:.1%}")
            print(f"Baseline Avg Reward:    {baseline_reward:.3f}")
            print(f"Success Improvement:    {improvement:+.1%}")
            print(f"Reward Improvement:     {reward_improvement:+.3f}")

            final_metrics.update({
                "final/baseline_success": baseline_success,
                "final/baseline_reward": baseline_reward,
                "final/success_improvement": improvement,
                "final/reward_improvement": reward_improvement
            })

        # Get final mediator statistics
        mediator_stats = tsc_agent.get_mediator_stats()
        mediator_avg_reward = mediator_stats.get('recent_avg_reward', 0)

        print(f"\nMediator Learning Progress:")
        print(f"Total Training Steps:      {mediator_stats.get('total_steps', 0)}")
        print(f"Current Ask Rate:          {mediator_stats.get('recent_ask_rate', 0):.3f}")
        print(f"Mediator Avg Reward:       {mediator_avg_reward:.3f}")  # ADDED BACK: Mediator's internal reward
        print(f"Interrupt Efficiency:      {mediator_stats.get('interrupt_efficiency', 0):.1%}")
        print(f"Recent Agreement Rate:     {mediator_stats.get('recent_agreement_rate', 0):.1%}")
        print(f"Final λ Penalty:           {mediator_stats.get('lambda_penalty', 0.02):.3f}")
        print(f"Final Agreement Penalty:   {mediator_stats.get('agreement_penalty', 0.1):.3f}")
        print(f"Training Phase:            {mediator_stats.get('training_phase', 'unknown')}")

        # ADDED: Explanation of the two different rewards
        print(f"\n📊 REWARD BREAKDOWN:")
        print(f"Task Reward:     {avg_reward:.3f} = Environment success/failure reward")
        print(f"Mediator Reward: {mediator_avg_reward:.3f} = Internal reward with agreement penalties")
        print(f"                 (Negative = too many unnecessary interrupts)")
        print(f"                 (Positive = efficient interrupt decisions)")

        final_metrics.update({
            "final/mediator_total_steps": mediator_stats.get('total_steps', 0),
            "final/mediator_ask_rate": mediator_stats.get('recent_ask_rate', 0),
            "final/mediator_avg_reward": mediator_stats.get('recent_avg_reward', 0),
            "final/mediator_efficiency": mediator_stats.get('interrupt_efficiency', 0),
            "final/mediator_agreement_rate": mediator_stats.get('recent_agreement_rate', 0),
            "final/mediator_lambda": mediator_stats.get('lambda_penalty', 0.02),
            "final/mediator_agreement_penalty": mediator_stats.get('agreement_penalty', 0.1),
        })

        wandb.log(final_metrics)

        # Save trained mediator
        save_path = "models/efficiency_focused_mediator_trained.pt"
        os.makedirs("models", exist_ok=True)
        tsc_agent.save_mediator(save_path)
        logger.info(f"Saved efficiency-focused trained mediator to {save_path}")

        # Save model as WandB artifact
        artifact = wandb.Artifact("efficiency_mediator_model", type="model")
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)

        # FIXED: Enhanced success criteria
        mediator_reward = mediator_stats.get('recent_avg_reward', 0)
        efficiency = mediator_stats.get('interrupt_efficiency', 0)

        if success_rate > 0.8 and efficiency > 0.5 and avg_interrupt_rate < 0.3:
            logger.info("🎉 PERFECT! High success, high efficiency, low interrupt rate!")
        elif success_rate > 0.7 and efficiency > 0.3:
            logger.info("✅ GOOD! Decent performance with reasonable efficiency")
        elif efficiency < 0.2:
            logger.warning("⚠️  EFFICIENCY PROBLEM: Too many unnecessary interrupts")
        else:
            logger.info("🔄 LEARNING: Mediator still improving efficiency")

    else:
        logger.error("No successful episodes completed!")

    # Cleanup
    rl_env.close()
    llm_env.close()
    wandb.finish()


if __name__ == '__main__':
    main()