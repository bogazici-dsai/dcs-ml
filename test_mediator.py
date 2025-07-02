import os
import torch
import numpy as np
import wandb
from loguru import logger
from typing import Dict, List

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


def evaluate_agent_simple(agent, test_env, rl_env, num_episodes: int = 20, agent_name: str = "Agent"):
    """
    Simple evaluation following the EXACT same pattern as training.
    Enhanced with detailed WandB logging for performance analysis.
    """
    logger.info(f"🧪 Testing {agent_name} over {num_episodes} episodes...")
    results = []

    # Initialize per-episode tracking for WandB
    episode_data = []

    for episode in range(num_episodes):
        # Reset both environments with same seed (like in training)
        test_obs, _ = test_env.reset(seed=episode + 1000)
        rl_obs, _ = rl_env.reset(seed=episode + 1000)

        done = False
        total_reward = 0
        steps = 0
        interrupts = 0
        overrides = 0
        agreements = 0
        step_rewards = []

        # Track step-by-step data for detailed analysis
        step_data = []

        while not done and steps < 100:
            steps += 1

            if hasattr(agent, 'agent_run'):
                # TSC Agent with mediator - EXACTLY like training
                try:
                    # RL AGENT MAKES PRIMARY DECISION (using RL env obs)
                    if hasattr(agent, 'rl_agent'):
                        rl_action, _ = agent.rl_agent.predict(rl_obs, deterministic=True)
                    else:
                        rl_action, _ = agent.predict(rl_obs, deterministic=True)
                    rl_action = int(rl_action)
                except Exception as e:
                    logger.warning(f"RL action prediction failed: {e}, using action 0")
                    rl_action = 0

                # MEDIATOR DECIDES (using test env obs - should be llm_env for TSC)
                env_info = {"env": "MiniGrid-DoorKey-6x6-v0", "llm_env": test_env}

                final_action, was_interrupted, info = agent.agent_run(
                    sim_step=steps,
                    obs=test_obs,  # Use test environment obs (llm_env for TSC)
                    rl_action=rl_action,
                    infos=env_info,
                    reward=None,  # No training
                    use_learned_asking=True
                )

                if was_interrupted:
                    interrupts += 1
                    if info.get('llm_plan_changed', False):
                        overrides += 1
                    else:
                        agreements += 1

                # Track step-level data
                step_data.append({
                    'step': steps,
                    'rl_action': rl_action,
                    'final_action': final_action,
                    'was_interrupted': was_interrupted,
                    'llm_plan_changed': info.get('llm_plan_changed', False) if was_interrupted else False,
                    'current_plan': info.get('current_plan', 'unknown') if was_interrupted else 'continuing'
                })

            else:
                # Baseline RL agent - simple case (test_env == rl_env)
                try:
                    final_action, _ = agent.predict(test_obs, deterministic=True)
                    final_action = int(final_action)
                except Exception as e:
                    logger.warning(f"Baseline prediction failed: {e}, using action 0")
                    final_action = 0

                # Track step-level data for baseline
                step_data.append({
                    'step': steps,
                    'final_action': final_action,
                    'was_interrupted': False,
                    'llm_plan_changed': False,
                    'current_plan': 'baseline_rl'
                })

            # EXECUTE ACTION IN BOTH ENVIRONMENTS (exactly like training)
            try:
                test_obs, reward, terminated, truncated, _ = test_env.step(final_action)
                if hasattr(agent, 'agent_run'):
                    # Also step RL environment to keep in sync (for TSC agent)
                    rl_obs, _, _, _, _ = rl_env.step(final_action)
            except Exception as e:
                logger.error(f"Environment step failed: {e}")
                test_obs, reward, terminated, truncated, _ = test_env.step(0)
                if hasattr(agent, 'agent_run'):
                    rl_obs, _, _, _, _ = rl_env.step(0)

            done = terminated or truncated
            total_reward += reward
            step_rewards.append(reward)

            # Add reward to current step data
            step_data[-1]['reward'] = reward
            step_data[-1]['cumulative_reward'] = total_reward

        success = total_reward > 0
        interrupt_rate = interrupts / steps if steps > 0 else 0
        efficiency = overrides / max(interrupts, 1) if interrupts > 0 else 1.0
        avg_step_reward = np.mean(step_rewards) if step_rewards else 0

        episode_result = {
            'episode': episode,
            'reward': total_reward,
            'steps': steps,
            'success': success,
            'interrupts': interrupts,
            'overrides': overrides,
            'agreements': agreements,
            'interrupt_rate': interrupt_rate,
            'efficiency': efficiency,
            'avg_step_reward': avg_step_reward,
            'step_data': step_data
        }

        results.append(episode_result)
        episode_data.append(episode_result)

        # Log individual episode to WandB
        episode_log = {
            f"episode_details/{agent_name.lower().replace(' ', '_')}/episode": episode,
            f"episode_details/{agent_name.lower().replace(' ', '_')}/reward": total_reward,
            f"episode_details/{agent_name.lower().replace(' ', '_')}/success": int(success),
            f"episode_details/{agent_name.lower().replace(' ', '_')}/steps": steps,
            f"episode_details/{agent_name.lower().replace(' ', '_')}/interrupts": interrupts,
            f"episode_details/{agent_name.lower().replace(' ', '_')}/overrides": overrides,
            f"episode_details/{agent_name.lower().replace(' ', '_')}/agreements": agreements,
            f"episode_details/{agent_name.lower().replace(' ', '_')}/efficiency": efficiency,
            f"episode_details/{agent_name.lower().replace(' ', '_')}/interrupt_rate": interrupt_rate,
            f"episode_details/{agent_name.lower().replace(' ', '_')}/avg_step_reward": avg_step_reward,
        }
        wandb.log(episode_log)

        # Progress update every 5 episodes with enhanced WandB logging
        if episode % 5 == 0 and episode > 0:
            recent = results[-5:]
            avg_success = np.mean([r['success'] for r in recent])
            avg_interrupts = np.mean([r['interrupts'] for r in recent])
            avg_efficiency = np.mean([r['efficiency'] for r in recent])
            avg_reward = np.mean([r['reward'] for r in recent])
            avg_overrides = np.mean([r['overrides'] for r in recent])
            avg_agreements = np.mean([r['agreements'] for r in recent])

            logger.info(f"{agent_name} Episode {episode}: Success={avg_success:.1%}, "
                        f"Reward={avg_reward:.3f}, Interrupts={avg_interrupts:.1f}, "
                        f"Efficiency={avg_efficiency:.1%}")

            # Enhanced progress logging to WandB
            progress_log = {
                f"progress/{agent_name.lower().replace(' ', '_')}/episode": episode,
                f"progress/{agent_name.lower().replace(' ', '_')}/recent_success_rate": avg_success,
                f"progress/{agent_name.lower().replace(' ', '_')}/recent_avg_reward": avg_reward,
                f"progress/{agent_name.lower().replace(' ', '_')}/recent_interrupts": avg_interrupts,
                f"progress/{agent_name.lower().replace(' ', '_')}/recent_efficiency": avg_efficiency,
                f"progress/{agent_name.lower().replace(' ', '_')}/recent_overrides": avg_overrides,
                f"progress/{agent_name.lower().replace(' ', '_')}/recent_agreements": avg_agreements,
                f"progress/{agent_name.lower().replace(' ', '_')}/recent_interrupt_rate": np.mean(
                    [r['interrupt_rate'] for r in recent]),
            }
            wandb.log(progress_log)

    return results


def create_detailed_analysis_tables(baseline_results, tsc_results):
    """Create detailed WandB tables for analysis."""

    # Episode-by-episode comparison table
    comparison_data = []
    for i in range(min(len(baseline_results), len(tsc_results))):
        baseline = baseline_results[i]
        tsc = tsc_results[i]
        comparison_data.append([
            baseline['episode'],
            baseline['success'], tsc['success'],
            baseline['reward'], tsc['reward'],
            baseline['steps'], tsc['steps'],
            baseline['interrupts'], tsc['interrupts'],
            baseline['efficiency'], tsc['efficiency']
        ])

    comparison_table = wandb.Table(
        columns=[
            "Episode",
            "Baseline Success", "TSC Success",
            "Baseline Reward", "TSC Reward",
            "Baseline Steps", "TSC Steps",
            "Baseline Interrupts", "TSC Interrupts",
            "Baseline Efficiency", "TSC Efficiency"
        ],
        data=comparison_data
    )

    # Performance distribution table
    baseline_stats = {
        'success_rate': np.mean([r['success'] for r in baseline_results]),
        'avg_reward': np.mean([r['reward'] for r in baseline_results]),
        'std_reward': np.std([r['reward'] for r in baseline_results]),
        'avg_steps': np.mean([r['steps'] for r in baseline_results]),
        'avg_interrupts': np.mean([r['interrupts'] for r in baseline_results]),
    }

    tsc_stats = {
        'success_rate': np.mean([r['success'] for r in tsc_results]),
        'avg_reward': np.mean([r['reward'] for r in tsc_results]),
        'std_reward': np.std([r['reward'] for r in tsc_results]),
        'avg_steps': np.mean([r['steps'] for r in tsc_results]),
        'avg_interrupts': np.mean([r['interrupts'] for r in tsc_results]),
        'avg_efficiency': np.mean([r['efficiency'] for r in tsc_results]),
    }

    stats_table = wandb.Table(
        columns=["Metric", "Baseline", "TSC + Mediator", "Improvement"],
        data=[
            ["Success Rate", f"{baseline_stats['success_rate']:.1%}",
             f"{tsc_stats['success_rate']:.1%}",
             f"{tsc_stats['success_rate'] - baseline_stats['success_rate']:+.1%}"],
            ["Average Reward", f"{baseline_stats['avg_reward']:.3f}",
             f"{tsc_stats['avg_reward']:.3f}",
             f"{tsc_stats['avg_reward'] - baseline_stats['avg_reward']:+.3f}"],
            ["Reward Std Dev", f"{baseline_stats['std_reward']:.3f}",
             f"{tsc_stats['std_reward']:.3f}",
             f"{tsc_stats['std_reward'] - baseline_stats['std_reward']:+.3f}"],
            ["Average Steps", f"{baseline_stats['avg_steps']:.1f}",
             f"{tsc_stats['avg_steps']:.1f}",
             f"{tsc_stats['avg_steps'] - baseline_stats['avg_steps']:+.1f}"],
            ["Average Interrupts", f"{baseline_stats['avg_interrupts']:.1f}",
             f"{tsc_stats['avg_interrupts']:.1f}",
             f"{tsc_stats['avg_interrupts'] - baseline_stats['avg_interrupts']:+.1f}"],
            ["Efficiency", "N/A", f"{tsc_stats['avg_efficiency']:.1%}", "N/A"]
        ]
    )

    return comparison_table, stats_table


def main():
    """Enhanced performance test for mediator with comprehensive WandB integration."""

    # Enhanced WandB initialization
    wandb.init(
        project="TSC_Mediator_Testing",
        entity="BILGEM_DCS_RL",
        config={
            "test_type": "performance_evaluation",
            "env_name": "MiniGrid-DoorKey-6x6-v0",
            "test_episodes": 15,
            "max_steps": 100,
            "llm_model": "llama3.1:8b",
            "evaluation_mode": "deterministic",
            "comparison_agents": ["baseline_rl", "tsc_mediator"],
            "test_date": "2025-07-02",
            "test_purpose": "mediator_performance_analysis"
        },
        tags=["testing", "performance", "mediator", "comparison"]
    )

    logger.info("🚀 Starting Enhanced Mediator Performance Test with WandB Integration")

    # Setup
    device = get_device()
    logger.info(f"Using device: {device}")

    # Log device info to WandB
    wandb.log({
        "setup/device": str(device),
        "setup/device_available": True
    })

    # Initialize LLM
    try:
        chat = ChatOllama(model="llama3.1:8b", temperature=0.1)
        llm = RunnableLambda(lambda x: chat.invoke(x))
        logger.info("✅ Llama initialized")
        wandb.log({"setup/llm_initialization": "success"})
    except Exception as e:
        logger.error(f"❌ LLM initialization failed: {e}")
        wandb.log({"setup/llm_initialization": "failed", "setup/llm_error": str(e)})
        return

    # Initialize environments
    env_name = "MiniGrid-DoorKey-6x6-v0"
    rl_env, llm_env = make_env(env_name=env_name, max_steps=100)
    wandb.log({
        "setup/environment": env_name,
        "setup/max_steps": 100
    })

    # Load RL Agent
    model_path = "models/ppo_minigrid_doorkey_6x6_250000_steps"
    try:
        rl_agent = PPO.load(model_path, device=device)
        logger.info("✅ RL agent loaded")
        wandb.log({
            "setup/rl_agent_loading": "success",
            "setup/rl_model_path": model_path
        })
    except Exception as e:
        logger.error(f"❌ RL agent loading failed: {e}")
        wandb.log({
            "setup/rl_agent_loading": "failed",
            "setup/rl_error": str(e)
        })
        return

    # Initialize TSC Agent
    obs_shape = llm_env.observation_space['image'].shape
    tsc_agent = TSCAgentWithMediator(
        llm=llm,
        obs_shape=obs_shape,
        device=device,
        verbose=False,  # Quiet for testing
        train_mediator=False  # No training during test
    )

    # Try to load trained mediator
    mediator_paths = [
        "models/efficiency_focused_mediator_trained.pt",
        "models/best_efficient_mediator_ep_*.pt"  # Pattern for best models
    ]

    mediator_loaded = False
    loaded_path = None
    for path in mediator_paths:
        if '*' in path:
            # Handle wildcard pattern
            import glob
            matching_files = glob.glob(path)
            if matching_files:
                # Get the most recent file
                path = max(matching_files, key=os.path.getctime)

        if os.path.exists(path):
            try:
                tsc_agent.load_mediator(path)
                logger.info(f"✅ Loaded mediator from {path}")
                mediator_loaded = True
                loaded_path = path
                break
            except Exception as e:
                logger.warning(f"⚠️ Could not load {path}: {e}")

    if not mediator_loaded:
        logger.info("📝 Using untrained mediator")

    # Enhanced mediator loading status logging
    wandb.log({
        "setup/mediator_loaded": mediator_loaded,
        "setup/mediator_source": "trained" if mediator_loaded else "untrained",
        "setup/mediator_path": loaded_path if loaded_path else "none"
    })

    # Get mediator stats for logging
    if mediator_loaded and hasattr(tsc_agent, 'get_mediator_stats'):
        mediator_stats = tsc_agent.get_mediator_stats()
        wandb.log({
            "setup/mediator_total_steps": mediator_stats.get('total_steps', 0),
            "setup/mediator_training_phase": mediator_stats.get('training_phase', 'unknown'),
            "setup/mediator_ask_rate": mediator_stats.get('recent_ask_rate', 0),
        })

    # Create wrappers
    class TSCWrapper:
        def __init__(self, rl_agent, tsc_agent):
            self.rl_agent = rl_agent
            self.tsc_agent = tsc_agent

        def predict(self, obs, deterministic=True):
            return self.rl_agent.predict(obs, deterministic)

        def agent_run(self, sim_step, obs, rl_action, infos, reward=None, use_learned_asking=True):
            return self.tsc_agent.agent_run(sim_step, obs, rl_action, infos, reward, use_learned_asking)

    tsc_wrapper = TSCWrapper(rl_agent, tsc_agent)

    # Run tests - EXACTLY like training setup
    num_episodes = 15  # Short test
    wandb.log({"test_config/num_episodes": num_episodes})

    logger.info("🔍 Testing Baseline RL...")
    # Baseline: RL agent on RL environment (both same)
    baseline_results = evaluate_agent_simple(rl_agent, rl_env, rl_env, num_episodes, "Baseline RL")

    logger.info("🧠 Testing TSC + Mediator...")
    # TSC: TSC agent on LLM environment, but RL actions from RL environment
    tsc_results = evaluate_agent_simple(tsc_wrapper, llm_env, rl_env, num_episodes, "TSC + Mediator")

    # Enhanced Analysis with WandB tables and visualizations
    logger.info("📊 Creating detailed analysis...")

    def analyze_detailed(results, name):
        """Enhanced analysis function with more metrics."""
        success_rate = np.mean([r['success'] for r in results])
        avg_reward = np.mean([r['reward'] for r in results])
        std_reward = np.std([r['reward'] for r in results])
        avg_interrupts = np.mean([r['interrupts'] for r in results])
        avg_efficiency = np.mean([r['efficiency'] for r in results])
        avg_interrupt_rate = np.mean([r['interrupt_rate'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        avg_overrides = np.mean([r['overrides'] for r in results])
        avg_agreements = np.mean([r['agreements'] for r in results])

        # Success consistency (how often does it succeed when it should)
        success_consistency = np.std([r['success'] for r in results])

        # Reward efficiency (reward per step)
        reward_per_step = avg_reward / max(avg_steps, 1)

        print(f"\n{name}:")
        print(f"  Success Rate:         {success_rate:.1%}")
        print(f"  Average Reward:       {avg_reward:.3f} ± {std_reward:.3f}")
        print(f"  Reward per Step:      {reward_per_step:.4f}")
        print(f"  Average Steps:        {avg_steps:.1f}")
        print(f"  Avg Interrupts:       {avg_interrupts:.1f}")
        print(f"  Avg Overrides:        {avg_overrides:.1f}")
        print(f"  Avg Agreements:       {avg_agreements:.1f}")
        print(f"  Interrupt Rate:       {avg_interrupt_rate:.1%}")
        print(f"  Efficiency:           {avg_efficiency:.1%}")
        print(f"  Success Consistency:  {success_consistency:.3f}")

        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_interrupts': avg_interrupts,
            'avg_efficiency': avg_efficiency,
            'avg_interrupt_rate': avg_interrupt_rate,
            'avg_steps': avg_steps,
            'avg_overrides': avg_overrides,
            'avg_agreements': avg_agreements,
            'success_consistency': success_consistency,
            'reward_per_step': reward_per_step
        }

    # Analysis
    print("\n" + "=" * 70)
    print("📊 ENHANCED PERFORMANCE TEST RESULTS")
    print("=" * 70)

    baseline_stats = analyze_detailed(baseline_results, "📊 Baseline RL")
    tsc_stats = analyze_detailed(tsc_results, "🧠 TSC + Mediator")

    # Create detailed WandB tables
    comparison_table, stats_table = create_detailed_analysis_tables(baseline_results, tsc_results)

    # Log tables to WandB
    wandb.log({
        "analysis/episode_comparison": comparison_table,
        "analysis/performance_summary": stats_table
    })

    # Comprehensive results logging to WandB
    comprehensive_results = {
        # Baseline results
        "final_results/baseline_success_rate": baseline_stats['success_rate'],
        "final_results/baseline_avg_reward": baseline_stats['avg_reward'],
        "final_results/baseline_std_reward": baseline_stats['std_reward'],
        "final_results/baseline_avg_steps": baseline_stats['avg_steps'],
        "final_results/baseline_avg_interrupts": baseline_stats['avg_interrupts'],
        "final_results/baseline_avg_efficiency": baseline_stats['avg_efficiency'],
        "final_results/baseline_interrupt_rate": baseline_stats['avg_interrupt_rate'],
        "final_results/baseline_reward_per_step": baseline_stats['reward_per_step'],
        "final_results/baseline_success_consistency": baseline_stats['success_consistency'],

        # TSC results
        "final_results/tsc_success_rate": tsc_stats['success_rate'],
        "final_results/tsc_avg_reward": tsc_stats['avg_reward'],
        "final_results/tsc_std_reward": tsc_stats['std_reward'],
        "final_results/tsc_avg_steps": tsc_stats['avg_steps'],
        "final_results/tsc_avg_interrupts": tsc_stats['avg_interrupts'],
        "final_results/tsc_avg_overrides": tsc_stats['avg_overrides'],
        "final_results/tsc_avg_agreements": tsc_stats['avg_agreements'],
        "final_results/tsc_avg_efficiency": tsc_stats['avg_efficiency'],
        "final_results/tsc_interrupt_rate": tsc_stats['avg_interrupt_rate'],
        "final_results/tsc_reward_per_step": tsc_stats['reward_per_step'],
        "final_results/tsc_success_consistency": tsc_stats['success_consistency'],
    }

    # Compare performance
    print(f"\n🎯 ENHANCED PERFORMANCE COMPARISON:")
    success_diff = tsc_stats['success_rate'] - baseline_stats['success_rate']
    reward_diff = tsc_stats['avg_reward'] - baseline_stats['avg_reward']
    efficiency_ratio = tsc_stats['avg_efficiency'] / max(baseline_stats['avg_efficiency'], 0.01)
    interrupt_overhead = tsc_stats['avg_interrupts'] - baseline_stats['avg_interrupts']

    print(f"  Success Rate Difference:  {success_diff:+.1%}")
    print(f"  Reward Difference:        {reward_diff:+.3f}")
    print(f"  Efficiency Ratio:         {efficiency_ratio:.2f}x")
    print(f"  Interrupt Overhead:       {interrupt_overhead:+.1f}")

    # Enhanced comparison metrics for WandB
    comparison_metrics = {
        "comparison/success_rate_improvement": success_diff,
        "comparison/reward_improvement": reward_diff,
        "comparison/efficiency_ratio": efficiency_ratio,
        "comparison/interrupt_overhead": interrupt_overhead,
        "comparison/reward_efficiency_improvement": (tsc_stats['reward_per_step'] - baseline_stats['reward_per_step']),
        "comparison/consistency_improvement": (
                    baseline_stats['success_consistency'] - tsc_stats['success_consistency']),
        # Lower is better for consistency
    }

    # Combine all results for final logging
    all_results = {**comprehensive_results, **comparison_metrics}
    wandb.log(all_results)

    # Get enhanced mediator internal stats
    if hasattr(tsc_wrapper.tsc_agent, 'get_mediator_stats'):
        mediator_stats = tsc_wrapper.tsc_agent.get_mediator_stats()
        print(f"\n🔧 ENHANCED MEDIATOR ANALYSIS:")
        print(f"  Ask Rate:               {mediator_stats.get('recent_ask_rate', 0):.3f}")
        print(f"  Agreement Rate:         {mediator_stats.get('recent_agreement_rate', 0):.1%}")
        print(f"  Override Rate:          {mediator_stats.get('recent_override_rate', 0):.1%}")
        print(f"  Interrupt Efficiency:   {mediator_stats.get('interrupt_efficiency', 0):.1%}")
        print(f"  Training Phase:         {mediator_stats.get('training_phase', 'unknown')}")

        # Enhanced mediator stats for WandB
        mediator_analysis = {
            "mediator_analysis/ask_rate": mediator_stats.get('recent_ask_rate', 0),
            "mediator_analysis/agreement_rate": mediator_stats.get('recent_agreement_rate', 0),
            "mediator_analysis/override_rate": mediator_stats.get('recent_override_rate', 0),
            "mediator_analysis/interrupt_efficiency": mediator_stats.get('interrupt_efficiency', 0),
            "mediator_analysis/total_training_steps": mediator_stats.get('total_steps', 0),
            "mediator_analysis/training_phase": 1 if mediator_stats.get('training_phase') == 'exploitation' else 0,
        }

        # Loop detection info if available
        if 'forced_rl_mode_steps' in mediator_stats:
            print(f"  Forced RL Steps:        {mediator_stats.get('forced_rl_mode_steps', 0)}")
            print(f"  Position Stuck Count:   {mediator_stats.get('position_stuck_count', 0)}")
            print(f"  Loop Detection Active:  {mediator_stats.get('loop_detection_active', False)}")

            mediator_analysis.update({
                "mediator_analysis/forced_rl_steps": mediator_stats.get('forced_rl_mode_steps', 0),
                "mediator_analysis/position_stuck_count": mediator_stats.get('position_stuck_count', 0),
                "mediator_analysis/loop_detection_active": mediator_stats.get('loop_detection_active', False),
            })

        wandb.log(mediator_analysis)

    # Enhanced final assessment with scoring
    print(f"\n🏆 ENHANCED FINAL ASSESSMENT:")

    # Create comprehensive scoring system
    performance_score = 0
    efficiency_score = 0
    consistency_score = 0

    # Performance scoring (0-4 points)
    if success_diff > 0.05:
        performance_score = 4
        perf_msg = "🎉 EXCELLENT! Significant performance improvement!"
    elif success_diff > 0:
        performance_score = 3
        perf_msg = "✅ GOOD! Performance improvement achieved!"
    elif tsc_stats['success_rate'] >= baseline_stats['success_rate'] * 0.95:
        performance_score = 2
        perf_msg = "📈 ACCEPTABLE! Performance maintained!"
    elif tsc_stats['success_rate'] >= baseline_stats['success_rate'] * 0.85:
        performance_score = 1
        perf_msg = "⚠️ MARGINAL! Some performance degradation!"
    else:
        performance_score = 0
        perf_msg = "❌ POOR! Significant performance loss!"

    # Efficiency scoring (0-4 points)
    if tsc_stats['avg_efficiency'] > 0.7:
        efficiency_score = 4
        eff_msg = "🚀 EXCELLENT EFFICIENCY! Very smart interrupts!"
    elif tsc_stats['avg_efficiency'] > 0.5:
        efficiency_score = 3
        eff_msg = "👍 GOOD EFFICIENCY! Reasonable interrupt decisions!"
    elif tsc_stats['avg_efficiency'] > 0.3:
        efficiency_score = 2
        eff_msg = "📊 MODERATE EFFICIENCY! Room for improvement!"
    elif tsc_stats['avg_efficiency'] > 0.1:
        efficiency_score = 1
        eff_msg = "⚠️ LOW EFFICIENCY! Many unnecessary interrupts!"
    else:
        efficiency_score = 0
        eff_msg = "❌ VERY LOW EFFICIENCY! Wasting too many interactions!"

    # Consistency scoring (0-2 points)
    if tsc_stats['success_consistency'] <= baseline_stats['success_consistency']:
        consistency_score = 2
        cons_msg = "🎯 CONSISTENT! Reliable performance!"
    elif tsc_stats['success_consistency'] <= baseline_stats['success_consistency'] * 1.5:
        consistency_score = 1
        cons_msg = "📊 MODERATE CONSISTENCY! Some variability!"
    else:
        consistency_score = 0
        cons_msg = "⚠️ INCONSISTENT! High performance variability!"

    total_score = performance_score + efficiency_score + consistency_score
    max_score = 10

    print(f"\n📊 DETAILED SCORING:")
    print(f"  Performance Score: {performance_score}/4 - {perf_msg}")
    print(f"  Efficiency Score:  {efficiency_score}/4 - {eff_msg}")
    print(f"  Consistency Score: {consistency_score}/2 - {cons_msg}")
    print(f"  TOTAL SCORE:       {total_score}/{max_score} ({total_score / max_score:.1%})")

    # Overall rating
    if total_score >= 8:
        overall_rating = "excellent"
        rating_emoji = "🏆"
        rating_msg = "EXCELLENT! Ready for deployment!"
    elif total_score >= 6:
        overall_rating = "good"
        rating_emoji = "✅"
        rating_msg = "GOOD! Solid performance with minor improvements needed!"
    elif total_score >= 4:
        overall_rating = "acceptable"
        rating_emoji = "📈"
        rating_msg = "ACCEPTABLE! Shows promise but needs improvement!"
    elif total_score >= 2:
        overall_rating = "poor"
        rating_emoji = "⚠️"
        rating_msg = "POOR! Significant improvements required!"
    else:
        overall_rating = "very_poor"
        rating_emoji = "❌"
        rating_msg = "VERY POOR! Back to the drawing board!"

    print(f"\n{rating_emoji} OVERALL RATING: {overall_rating.upper()} - {rating_msg}")

    # Log comprehensive scoring to WandB
    scoring_results = {
        "scoring/performance_score": performance_score,
        "scoring/efficiency_score": efficiency_score,
        "scoring/consistency_score": consistency_score,
        "scoring/total_score": total_score,
        "scoring/max_score": max_score,
        "scoring/overall_percentage": total_score / max_score,
        "scoring/overall_rating": overall_rating,
        "scoring/rating_numeric": total_score,  # For easy sorting/filtering
    }
    wandb.log(scoring_results)

    # Create comprehensive performance summary
    summary_table = wandb.Table(
        columns=["Category", "Score", "Max", "Percentage", "Assessment"],
        data=[
            ["Performance", performance_score, 4, f"{performance_score / 4:.1%}",
             perf_msg.split("!")[-2] + "!" if "!" in perf_msg else perf_msg],
            ["Efficiency", efficiency_score, 4, f"{efficiency_score / 4:.1%}",
             eff_msg.split("!")[-2] + "!" if "!" in eff_msg else eff_msg],
            ["Consistency", consistency_score, 2, f"{consistency_score / 2:.1%}",
             cons_msg.split("!")[-2] + "!" if "!" in cons_msg else cons_msg],
            ["OVERALL", total_score, max_score, f"{total_score / max_score:.1%}", rating_msg]
        ]
    )
    wandb.log({"analysis/scoring_summary": summary_table})

    # Save enhanced report
    report_path = "enhanced_test_results.txt"
    with open(report_path, 'w') as f:
        f.write("ENHANCED MEDIATOR PERFORMANCE TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Configuration:\n")
        f.write(f"  Episodes Tested: {num_episodes}\n")
        f.write(f"  Mediator Loaded: {mediator_loaded}\n")
        f.write(f"  Model Path: {loaded_path if loaded_path else 'None'}\n\n")

        f.write(f"Baseline RL Performance:\n")
        f.write(f"  Success Rate: {baseline_stats['success_rate']:.1%}\n")
        f.write(f"  Average Reward: {baseline_stats['avg_reward']:.3f} ± {baseline_stats['std_reward']:.3f}\n")
        f.write(f"  Average Steps: {baseline_stats['avg_steps']:.1f}\n")
        f.write(f"  Reward per Step: {baseline_stats['reward_per_step']:.4f}\n\n")

        f.write(f"TSC + Mediator Performance:\n")
        f.write(f"  Success Rate: {tsc_stats['success_rate']:.1%}\n")
        f.write(f"  Average Reward: {tsc_stats['avg_reward']:.3f} ± {tsc_stats['std_reward']:.3f}\n")
        f.write(f"  Average Steps: {tsc_stats['avg_steps']:.1f}\n")
        f.write(f"  Interrupts: {tsc_stats['avg_interrupts']:.1f}\n")
        f.write(f"  Overrides: {tsc_stats['avg_overrides']:.1f}\n")
        f.write(f"  Agreements: {tsc_stats['avg_agreements']:.1f}\n")
        f.write(f"  Efficiency: {tsc_stats['avg_efficiency']:.1%}\n")
        f.write(f"  Reward per Step: {tsc_stats['reward_per_step']:.4f}\n\n")

        f.write(f"Improvements:\n")
        f.write(f"  Success Rate: {success_diff:+.1%}\n")
        f.write(f"  Average Reward: {reward_diff:+.3f}\n")
        f.write(f"  Efficiency Ratio: {efficiency_ratio:.2f}x\n")
        f.write(f"  Interrupt Overhead: {interrupt_overhead:+.1f}\n\n")

        f.write(f"Scoring:\n")
        f.write(f"  Performance: {performance_score}/4\n")
        f.write(f"  Efficiency: {efficiency_score}/4\n")
        f.write(f"  Consistency: {consistency_score}/2\n")
        f.write(f"  Total: {total_score}/{max_score} ({total_score / max_score:.1%})\n")
        f.write(f"  Overall Rating: {overall_rating.upper()}\n\n")
        f.write(f"Assessment: {rating_msg}\n")

    logger.info(f"📄 Enhanced report saved to {report_path}")

    # Save model as WandB artifact if mediator was loaded
    if mediator_loaded and loaded_path:
        try:
            artifact = wandb.Artifact(
                name="tested_mediator_model",
                type="model",
                metadata={
                    "test_score": total_score,
                    "success_rate": tsc_stats['success_rate'],
                    "efficiency": tsc_stats['avg_efficiency'],
                    "test_episodes": num_episodes,
                    "model_path": loaded_path
                }
            )
            artifact.add_file(loaded_path)
            wandb.log_artifact(artifact)
            logger.info("📦 Model artifact saved to WandB")
        except Exception as e:
            logger.warning(f"⚠️ Could not save model artifact: {e}")

    # Create efficiency vs performance scatter plot data
    efficiency_vs_performance = []
    for i, result in enumerate(tsc_results):
        efficiency_vs_performance.append([
            result['efficiency'],
            result['success'],
            result['reward'],
            result['episode']
        ])

    scatter_table = wandb.Table(
        columns=["Efficiency", "Success", "Reward", "Episode"],
        data=efficiency_vs_performance
    )
    wandb.log({"analysis/efficiency_vs_performance": scatter_table})

    # Log step-by-step analysis for first few episodes (detailed debugging)
    if tsc_results and len(tsc_results) > 0:
        for ep_idx in range(min(3, len(tsc_results))):  # First 3 episodes
            episode_steps = tsc_results[ep_idx].get('step_data', [])
            if episode_steps:
                step_analysis = []
                for step in episode_steps:
                    step_analysis.append([
                        ep_idx,
                        step['step'],
                        step['final_action'],
                        step['was_interrupted'],
                        step['llm_plan_changed'],
                        step['reward'],
                        step['cumulative_reward'],
                        step.get('current_plan', 'unknown')
                    ])

                step_table = wandb.Table(
                    columns=["Episode", "Step", "Action", "Interrupted", "Plan_Changed",
                             "Step_Reward", "Cumulative_Reward", "Current_Plan"],
                    data=step_analysis
                )
                wandb.log({f"detailed_analysis/episode_{ep_idx}_steps": step_table})

    # Cleanup
    rl_env.close()
    llm_env.close()

    # Final WandB summary
    wandb.summary.update({
        "final_score": total_score,
        "final_rating": overall_rating,
        "success_improvement": success_diff,
        "efficiency_achieved": tsc_stats['avg_efficiency'],
        "mediator_loaded": mediator_loaded,
        "test_completed": True
    })

    logger.info("🏁 Enhanced performance test complete!")

    return {
        'baseline_success': baseline_stats['success_rate'],
        'tsc_success': tsc_stats['success_rate'],
        'improvement': success_diff,
        'efficiency': tsc_stats['avg_efficiency'],
        'rating': overall_rating,
        'rating_score': total_score,
        'max_score': max_score,
        'performance_score': performance_score,
        'efficiency_score': efficiency_score,
        'consistency_score': consistency_score
    }


if __name__ == '__main__':
    try:
        results = main()

        # Enhanced final summary
        print(f"\n📋 FINAL SUMMARY:")
        print(f"   Success Rate: {results['tsc_success']:.1%} ({results['improvement']:+.1%} vs baseline)")
        print(f"   Efficiency: {results['efficiency']:.1%}")
        print(
            f"   Overall Score: {results['rating_score']}/{results['max_score']} ({results['rating_score'] / results['max_score']:.1%})")
        print(f"   Rating: {results['rating'].upper()}")

        # Component scores
        print(f"\n🔍 COMPONENT SCORES:")
        print(f"   Performance: {results['performance_score']}/4")
        print(f"   Efficiency: {results['efficiency_score']}/4")
        print(f"   Consistency: {results['consistency_score']}/2")

        # Final emoji summary
        if results['rating_score'] >= 8:
            print("🎉 SUCCESS: Mediator is performing excellently!")
        elif results['rating_score'] >= 6:
            print("👍 GOOD: Mediator shows solid performance!")
        elif results['rating_score'] >= 4:
            print("📈 PROGRESS: Mediator shows promise but needs work!")
        else:
            print("🔧 NEEDS WORK: Mediator requires significant improvement!")

        print(f"🔗 Check detailed results and visualizations in WandB dashboard")

        # Finish WandB run
        wandb.finish()

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        wandb.finish(exit_code=1)