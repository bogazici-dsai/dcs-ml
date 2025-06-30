import torch
import numpy as np
from stable_baselines3 import PPO
from utils.make_tsc_env import make_env


def diagnose_ppo_model():
    """Diagnose what's wrong with the PPO model"""
    print("=== PPO MODEL DIAGNOSIS ===")

    # Load model and environment
    model_path = "models/ppo_minigrid_doorkey_6x6_250000_steps"
    env_name = "MiniGrid-DoorKey-6x6-v0"

    try:
        model = PPO.load(model_path)
        print(f"✅ PPO model loaded successfully")

        rl_env, llm_env = make_env(env_name=env_name, max_steps=100)
        print(f"✅ Environment loaded successfully")

        # Check action space
        print(f"🔍 Action space: {rl_env.action_space}")
        print(f"🔍 Action space size: {rl_env.action_space.n}")

        # Test PPO for multiple episodes
        print("\n=== TESTING PPO ACTIONS ===")

        for episode in range(3):
            obs, _ = rl_env.reset(seed=episode)
            print(f"\nEpisode {episode}:")

            action_counts = {}
            for step in range(20):  # Just test first 20 steps
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

                action_counts[action] = action_counts.get(action, 0) + 1

                obs, reward, terminated, truncated, info = rl_env.step(action)

                if step < 5:  # Show first 5 actions
                    print(f"  Step {step}: Action {action}")

                if terminated or truncated:
                    print(f"  Episode ended at step {step}, Reward: {reward}")
                    break

            print(f"  Action distribution: {action_counts}")

            if len(set(action_counts.keys())) == 1:
                print("  ⚠️  WARNING: PPO is stuck on one action!")

        rl_env.close()
        llm_env.close()

    except Exception as e:
        print(f"❌ Error: {e}")


def create_simple_mediator_test():
    """Create a simple test for mediator with forced asking"""
    print("\n=== SIMPLE MEDIATOR TEST ===")

    from langchain_ollama import ChatOllama
    from langchain_core.runnables import RunnableLambda

    # Simple mediator that asks every 5 steps
    class SimpleMediatorTest:
        def __init__(self, llm):
            self.llm = llm
            self.step_count = 0

        def should_ask(self, obs, action):
            self.step_count += 1

            # Force ask every 5 steps OR if action is invalid
            if self.step_count % 5 == 1:
                return True, "Periodic ask (every 5 steps)"

            # Ask if action is invalid (6 = Done, which is forbidden)
            if action == 6:
                return True, "Invalid action detected (Action 6 = Done)"

            # Ask if action is repeated many times
            return False, "PPO action seems OK"

        def get_llm_action(self, obs, ppo_action):
            # Simple prompt for testing
            prompt = f"""
You are controlling a MiniGrid agent. PPO suggested action {ppo_action}.

Actions:
0: Turn left
1: Turn right  
2: Move forward
3: Pick up key
4: Drop key (FORBIDDEN)
5: Toggle door
6: Done (FORBIDDEN)

If PPO action is 4 or 6, override with action 0.
Otherwise, agree with PPO.

Respond with: Selected action: <number>
"""
            try:
                response = self.llm.invoke(prompt)
                text = response.content if hasattr(response, 'content') else str(response)

                # Extract action
                import re
                match = re.search(r"Selected\s*action\s*[:=]?\s*(\d)", text, re.IGNORECASE)
                if match:
                    return int(match.group(1))
                else:
                    print(f"⚠️  Couldn't parse LLM response: {text}")
                    return ppo_action

            except Exception as e:
                print(f"❌ LLM Error: {e}")
                return ppo_action

    # Test the simple mediator
    try:
        chat = ChatOllama(model="llama3", temperature=0.0)
        llm = RunnableLambda(lambda x: chat.invoke(x))

        mediator = SimpleMediatorTest(llm)
        model = PPO.load("models/ppo_minigrid_doorkey_6x6_250000_steps")
        rl_env, llm_env = make_env("MiniGrid-DoorKey-6x6-v0", max_steps=50)

        print("Testing simple mediator...")

        obs, _ = rl_env.reset(seed=42)
        total_reward = 0
        asks = 0
        overrides = 0

        for step in range(50):
            ppo_action, _ = model.predict(obs, deterministic=True)
            ppo_action = int(ppo_action)

            should_ask, reason = mediator.should_ask(obs, ppo_action)

            if should_ask:
                asks += 1
                llm_action = mediator.get_llm_action(obs, ppo_action)
                final_action = llm_action
                if llm_action != ppo_action:
                    overrides += 1
                    print(f"Step {step}: OVERRIDE PPO {ppo_action} → LLM {llm_action} ({reason})")
                else:
                    print(f"Step {step}: AGREE with PPO {ppo_action} ({reason})")
            else:
                final_action = ppo_action
                print(f"Step {step}: PPO {ppo_action} (no ask)")

            obs, reward, terminated, truncated, info = rl_env.step(final_action)
            total_reward += reward

            if terminated or truncated:
                print(f"Episode ended! Reward: {total_reward}")
                break

        print(f"\nResults:")
        print(f"Total asks: {asks}")
        print(f"Total overrides: {overrides}")
        print(f"Final reward: {total_reward}")

        rl_env.close()
        llm_env.close()

    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    # Run diagnostics
    diagnose_ppo_model()

    # Test simple mediator
    create_simple_mediator_test()