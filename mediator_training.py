import os
import torch
import numpy as np
from loguru import logger
from typing import Dict

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from TSCAssistant.tsc_assistant_mediator import TSCAgentWithMediator
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
    llm_obs, _ = llm_env.reset(seed=episode)

    done = False
    sim_step = 0
    total_reward = 0
    interrupts = 0
    overrides = 0

    logger.info(f"Episode {episode} START - RL is primary agent")

    while not done and sim_step < max_steps:
        sim_step += 1

        # STEP 1: RL AGENT (PPO) MAKES PRIMARY DECISION
        rl_action, _ = rl_agent.predict(rl_obs, deterministic=True)
        rl_action = int(rl_action)

        # STEP 2: MEDIATOR DECIDES - Should we interrupt RL with LLM?
        final_action, was_interrupted, interaction_info = tsc_agent.agent_run(
            sim_step=sim_step,
            obs=llm_obs,
            rl_action=rl_action,
            infos={"env": "MiniGrid-DoorKey-6x6-v0"},
            reward=total_reward,
            use_learned_asking=True
        )

        # Track statistics
        if was_interrupted:
            interrupts += 1
            if interaction_info.get('llm_plan_changed', False):
                overrides += 1

        # STEP 3: EXECUTE ACTION IN ENVIRONMENT
        rl_obs, reward, terminated, truncated, _ = rl_env.step(final_action)
        llm_obs, _, _, _, _ = llm_env.step(final_action)

        done = terminated or truncated
        total_reward += reward

    success_emoji = "✅" if total_reward > 0 else "❌"
    logger.info(f"Episode {episode} END: Reward={total_reward:.2f}, "
                f"Interrupts={interrupts}, Overrides={overrides}, Success={success_emoji}")

    return {
        'reward': total_reward,
        'steps': sim_step,
        'success': total_reward > 0,
        'interrupts': interrupts,
        'overrides': overrides,
        'interrupt_rate': interrupts / sim_step if sim_step > 0 else 0
    }


def main():
    """Main experiment with mediator flow."""
    logger.info("Starting Mediator Experiment")
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
    rl_agent = PPO.load(model_path, device=device)

    # Initialize TSC Agent with Mediator
    obs_shape = llm_env.observation_space['image'].shape
    tsc_agent = TSCAgentWithMediator(
        llm=llm,
        obs_shape=obs_shape,
        device=device,
        verbose=True,
        train_mediator=True
    )

    logger.info("Setup complete - RL primary, mediator learns when to interrupt")

    # Train mediator for 50 episodes (quick test)
    logger.info("Training mediator to learn when RL needs LLM help...")

    results = []
    for episode in range(50):
        result = run_episode_flow(
            rl_agent=rl_agent,
            tsc_agent=tsc_agent,
            rl_env=rl_env,
            llm_env=llm_env,
            episode=episode
        )
        results.append(result)

        # Log progress every 10 episodes
        if episode % 10 == 0:
            recent_results = results[-10:] if len(results) >= 10 else results
            avg_success = np.mean([r['success'] for r in recent_results])
            avg_interrupts = np.mean([r['interrupts'] for r in recent_results])
            avg_interrupt_rate = np.mean([r['interrupt_rate'] for r in recent_results])

            logger.info(f"Episode {episode}: Success={avg_success:.1%}, "
                        f"Avg_Interrupts={avg_interrupts:.1f}, "
                        f"Interrupt_Rate={avg_interrupt_rate:.1%}")

    # Print final results
    print("\n" + "=" * 60)
    print("MEDIATOR TRAINING RESULTS")
    print("=" * 60)
    success_rate = np.mean([r['success'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])
    avg_interrupts = np.mean([r['interrupts'] for r in results])
    avg_interrupt_rate = np.mean([r['interrupt_rate'] for r in results])

    print(f"Success Rate:        {success_rate:.1%}")
    print(f"Average Reward:      {avg_reward:.2f}")
    print(f"Average Interrupts:  {avg_interrupts:.1f} per episode")
    print(f"Average Interrupt Rate: {avg_interrupt_rate:.1%}")

    # Get mediator statistics
    mediator_stats = tsc_agent.get_mediator_stats()
    print(f"\nMediator Learning:")
    print(f"Recent Ask Rate:     {mediator_stats.get('recent_ask_rate', 0):.2f}")
    print(f"Interaction Efficiency: {mediator_stats.get('interaction_efficiency', 0):.2f}")

    # Save trained mediator
    save_path = "models/mediator_trained.pt"
    os.makedirs("models", exist_ok=True)
    tsc_agent.save_mediator(save_path)
    logger.info("Saved trained mediator")

    # Cleanup
    rl_env.close()
    llm_env.close()

    if success_rate > 0.5:
        logger.info("SUCCESS! Mediator learned when to interrupt RL effectively!")
    else:
        logger.info("Mediator still learning - may need more training episodes")


if __name__ == '__main__':
    main()