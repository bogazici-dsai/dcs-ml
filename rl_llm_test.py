# rl_llm_test.py
import os
import torch
import numpy as np

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from MinigridAssistant.minigrid_assistant import MinigridAgent
from utils.make_minigrid_env import make_env
from stable_baselines3 import PPO
import wandb

# ====== Utility ======
def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.backends.cuda.is_built() and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def print_summary(rewards, steps):
    print("\n✅ Evaluation Summary")
    print(f"Avg Reward: {np.mean(rewards):.2f}")
    print(f"Avg Steps: {np.mean(steps):.1f}")
    print(f"Success Rate: {np.mean([r > 0 for r in rewards]) * 100:.1f}%")

# ====== Main Evaluation Loop ======
if __name__ == '__main__':
    # Setup
    device = get_device()
    print("Using device:", device)

    llm_model_name = "llama3.1:8b"
    chat = ChatOllama(model=llm_model_name, temperature=0.0)
    llm = RunnableLambda(lambda x: chat.invoke(x))

    env_name = "Harfang"

    env = make_env(max_steps=5000)

    rl_env,llm_env = make_env(env_name=env_name, max_steps=100)


    model_path = "models/ppo_minigrid_doorkey_6x6_250000_steps_yedek"
    model = PPO.load(model_path, device=device)
    tsc_agent = MinigridAgent(llm=llm, verbose=True)

    # Evaluation settings
    num_episodes = 100
    llm_frequency = 4 # every n steps LLM refines action

    all_rewards = []
    all_steps = []
    wandb.init(
        project="PPO_MiniGrid_Training",
        entity="BILGEM_DCS_RL",
        name=f"ppo_minigrid_doorkey_eval_rl_llm_{num_episodes}_episodes",
        config={
            "env_name": env_name,
            "model_path": model_path,
            "num_episodes": num_episodes,
            "llm_model": llm_model_name,
            "llm_frequency": llm_frequency
        }
    )
    successes = []
    for episode in range(num_episodes):
        RL_obs, info_rl = rl_env.reset(seed=episode)
        LLM_obs, info_llm = llm_env.reset(seed=episode)
        done = False
        sim_step = 1
        total_reward = 0

        while not done:
            action, _ = model.predict(RL_obs, deterministic=True)

            if sim_step % llm_frequency == 2:
                action, _ = tsc_agent.agent_run(
                    sim_step=sim_step,
                    obs=LLM_obs,
                    action=action,
                    infos={"env": env_name,
                           "llm_env": llm_env}
                )

            RL_obs, reward, terminated, truncated, info = rl_env.step(action)
            LLM_obs, _, _, _, _ = llm_env.step(action)
            done = terminated or truncated
            total_reward += reward
            sim_step += 1
            llm_env.render()

        all_rewards.append(total_reward)
        all_steps.append(sim_step)
        is_success = total_reward > 0
        successes.append(is_success)
        cumulative_avg_success = np.mean(successes) * 100
        print(f"[Episode {episode+1}] Reward: {total_reward:.2f}, Steps: {sim_step}")
        wandb.log({
            "episode": episode + 1,
            "episode_reward": total_reward,
            "episode_steps": sim_step,
            "cumulative_avg_success_rate": cumulative_avg_success
        })
    llm_env.close()
    rl_env.close()
    wandb.log({
        "average_reward": np.mean(all_rewards),
        "average_steps": np.mean(all_steps),
        "success_rate": np.mean(successes) * 100
    })
    wandb.finish()

    print_summary(all_rewards, all_steps)
