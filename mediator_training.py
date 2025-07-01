import os
import torch
import numpy as np
from loguru import logger
from typing import Dict

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
# CHANGED: Import the enhanced TSC agent instead of the old one
from TSCAssistant.tsc_assistant_mediator import TSCAgentWithMediator  # This should be your enhanced version
from utils.make_tsc_env import make_env
from stable_baselines3 import PPO


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
    EPISODE FLOW: RL is primary, mediator decides when to interrupt with LLM
    """
    # Reset environments
    rl_obs, _ = rl_env.reset(seed=episode)
    llm_obs, llm_info = llm_env.reset(seed=episode)

    done = False
    sim_step = 0
    total_reward = 0
    interrupts = 0
    overrides = 0

    # ADDED: Store environment reference for has_key detection
    llm_info["llm_env"] = llm_env

    logger.info(f"Episode {episode} START - RL is primary agent")

    while not done and sim_step < max_steps:
        sim_step += 1

        # STEP 1: RL AGENT (PPO) MAKES PRIMARY DECISION
        rl_action, _ = rl_agent.predict(rl_obs, deterministic=True)
        rl_action = int(rl_action)

        # STEP 2: MEDIATOR DECIDES - Should we interrupt RL with LLM?
        # ENHANCED: Pass the llm_env info for better feature extraction
        final_action, was_interrupted, interaction_info = tsc_agent.agent_run(
            sim_step=sim_step,
            obs=llm_obs,
            rl_action=rl_action,
            infos={"env": "MiniGrid-DoorKey-6x6-v0", "llm_env": llm_env},  # ADDED llm_env
            reward=total_reward,
            use_learned_asking=True
        )

        # Track statistics
        if was_interrupted:
            interrupts += 1
            if interaction_info.get('llm_plan_changed', False):
                overrides += 1

        # STEP 3: EXECUTE ACTION IN ENVIRONMENT
        try:
            rl_obs, reward, terminated, truncated, _ = rl_env.step(final_action)
            llm_obs, _, _, _, llm_info = llm_env.step(final_action)

            # Update llm_env reference for next iteration
            llm_info["llm_env"] = llm_env

        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            # Use safe fallback action
            rl_obs, reward, terminated, truncated, _ = rl_env.step(0)  # Turn left
            llm_obs, _, _, _, llm_info = llm_env.step(0)
            llm_info["llm_env"] = llm_env

        done = terminated or truncated
        total_reward += reward

        # ADDED: Pass reward to mediator for better training
        if sim_step > 1:  # Skip first step since we don't have previous reward
            # Train mediator with the reward from this step
            tsc_agent.mediator.train_asking_policy(
                obs=llm_obs,
                action=rl_action,
                reward=reward,
                next_obs=llm_obs,
                asked_llm=was_interrupted,
                llm_plan_changed=interaction_info.get('llm_plan_changed', False)
            )

    success_emoji = "✅" if total_reward > 0 else "❌"
    logger.info(f"Episode {episode} END: Reward={total_reward:.2f}, "
                f"Interrupts={interrupts}, Overrides={overrides}, Success={success_emoji}")

    return {
        'reward': total_reward,
        'steps': sim_step,
        'success': total_reward > 0,
        'interrupts': interrupts,
        'overrides': overrides,
        'interrupt_rate': interrupts / sim_step if sim_step > 0 else 0,
        'override_rate': overrides / max(interrupts, 1)  # ADDED: override efficiency
    }


def evaluate_baseline(rl_agent, rl_env, num_episodes: int = 10):
    """
    ADDED: Evaluate baseline RL performance without LLM
    """
    logger.info("Evaluating baseline RL performance (no LLM)...")
    baseline_results = []

    for episode in range(num_episodes):
        obs, _ = rl_env.reset(seed=episode + 1000)  # Different seeds
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
    logger.info(f"Baseline RL Success Rate: {baseline_success:.1%}")
    return baseline_success


def main():
    """Main experiment with mediator flow."""
    logger.info("Starting Enhanced Mediator Experiment")
    logger.info("RL is PRIMARY, LLM interrupts when mediator decides useful")

    # Setup
    device = get_device()
    logger.info(f"Using device: {device}")

    # Initialize LLM
    chat = ChatOllama(model="llama3", temperature=0.0)
    llm = RunnableLambda(lambda x: chat.invoke(x))

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

    # Initialize Enhanced TSC Agent with Mediator
    obs_shape = llm_env.observation_space['image'].shape
    tsc_agent = TSCAgentWithMediator(
        llm=llm,
        obs_shape=obs_shape,
        device=device,
        verbose=True,  # Set to False for less logging
        train_mediator=True
    )

    logger.info("Setup complete - Enhanced TSC agent with rich spatial features")

    # ADDED: Baseline evaluation if RL agent available
    baseline_success = None
    if rl_agent is not None:
        baseline_success = evaluate_baseline(rl_agent, rl_env, num_episodes=5)

    # Train mediator
    num_episodes = 5  # INCREASED for better learning
    logger.info(f"Training mediator for {num_episodes} episodes...")

    results = []
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

            # Log progress every 20 episodes
            if episode % 20 == 0 and episode > 0:
                recent_results = results[-20:] if len(results) >= 20 else results
                avg_success = np.mean([r['success'] for r in recent_results])
                avg_interrupts = np.mean([r['interrupts'] for r in recent_results])
                avg_interrupt_rate = np.mean([r['interrupt_rate'] for r in recent_results])
                avg_override_rate = np.mean([r['override_rate'] for r in recent_results])

                logger.info(f"Episode {episode}: Success={avg_success:.1%}, "
                            f"Interrupts={avg_interrupts:.1f}, "
                            f"Interrupt_Rate={avg_interrupt_rate:.1%}, "
                            f"Override_Rate={avg_override_rate:.1%}")

                # Get mediator learning stats
                mediator_stats = tsc_agent.get_mediator_stats()
                logger.info(f"Mediator: Ask_Rate={mediator_stats.get('recent_ask_rate', 0):.2f}, "
                            f"Training_Phase={mediator_stats.get('training_phase', 'unknown')}")

        except Exception as e:
            logger.error(f"Episode {episode} failed: {e}")
            continue

    # Print final results
    print("\n" + "=" * 70)
    print("ENHANCED MEDIATOR TRAINING RESULTS")
    print("=" * 70)

    if results:
        success_rate = np.mean([r['success'] for r in results])
        avg_reward = np.mean([r['reward'] for r in results])
        avg_interrupts = np.mean([r['interrupts'] for r in results])
        avg_interrupt_rate = np.mean([r['interrupt_rate'] for r in results])
        avg_override_rate = np.mean([r['override_rate'] for r in results])

        print(f"Enhanced TSC Agent Performance:")
        print(f"Success Rate:           {success_rate:.1%}")
        print(f"Average Reward:         {avg_reward:.2f}")
        print(f"Average Interrupts:     {avg_interrupts:.1f} per episode")
        print(f"Average Interrupt Rate: {avg_interrupt_rate:.1%}")
        print(f"Average Override Rate:  {avg_override_rate:.1%} (when interrupted)")

        if baseline_success is not None:
            improvement = success_rate - baseline_success
            print(f"\nComparison to Baseline RL:")
            print(f"Baseline Success Rate:  {baseline_success:.1%}")
            print(f"Improvement:            {improvement:+.1%}")

        # Get final mediator statistics
        mediator_stats = tsc_agent.get_mediator_stats()
        print(f"\nMediator Learning Progress:")
        print(f"Total Training Steps:   {mediator_stats.get('total_steps', 0)}")
        print(f"Current Ask Rate:       {mediator_stats.get('recent_ask_rate', 0):.2f}")
        print(f"Training Phase:         {mediator_stats.get('training_phase', 'unknown')}")
        print(f"Interaction Efficiency: {mediator_stats.get('interaction_efficiency', 0):.2f}")

        # Save trained mediator
        save_path = "models/enhanced_mediator_trained.pt"
        os.makedirs("models", exist_ok=True)
        tsc_agent.save_mediator(save_path)
        logger.info(f"Saved enhanced trained mediator to {save_path}")

        # ADDED: Success criteria
        if success_rate > 0.7:
            logger.info("🎉 SUCCESS! Enhanced mediator learned effective LLM interruption!")
        elif success_rate > 0.5:
            logger.info("✅ GOOD! Mediator showing improvement, may benefit from more training")
        else:
            logger.info("⚠️  Mediator still learning - needs more training episodes")

        # ADDED: Efficiency analysis
        if avg_interrupt_rate < 0.3 and success_rate > 0.6:
            logger.info("🚀 EXCELLENT! Efficient mediator - high success with low interrupts")
        elif avg_interrupt_rate > 0.7:
            logger.info("💡 Note: High interrupt rate - mediator may be over-asking")

    else:
        logger.error("No successful episodes completed!")

    # Cleanup
    rl_env.close()
    llm_env.close()


if __name__ == '__main__':
    main()