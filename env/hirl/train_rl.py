from agents.TD3 import Agent as TD3Agent
from agents.SAC.agent import SacAgent as SACAgent
from utils.plot import plot_3d_trajectories, plot_distance
from utils.buffer import *
from utils.seed import *
from environments.HarfangEnv_GYM import *
import environments.dogfight_client as df

# Import your rule-based PID agent
from agents.rule_agent import RuleAgent

import numpy as np
import time
import math
from statistics import mean, pstdev
import datetime
import os
import csv
import argparse
import yaml
import gym
from torch.utils.tensorboard import SummaryWriter

# WandB imports
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING WandB not available. Install with: pip install wandb")


def setup_wandb(config, agent_name, env_type, model_name):
    """Setup WandB logging"""
    if not WANDB_AVAILABLE:
        return None

    try:
        wandb.init(
            entity="BILGEM_DCS_RL",
            project="pure-rl-ucav-dogfight",
            name=f"{agent_name}_{env_type}_{model_name}",
            config={
                "agent": agent_name,
                "environment": env_type,
                "model_name": model_name,
                "training_episodes": config.get('trainingEpisodes', 6000),
                "exploration_episodes": config.get('explorationEpisodes', 100),
                "max_steps": config.get('maxStep', 1500),
                "buffer_size": config.get('bufferSize', 10 ** 5),
                "batch_size": config.get('batchSize', 128),
                "actor_lr": config.get('actorLR', 1e-3),
                "critic_lr": config.get('criticLR', 1e-3),
                "gamma": config.get('gamma', 0.99),
                "tau": config.get('tau', 0.005),
            },
            tags=[agent_name, env_type, "pure_rl", "harfang3d"]
        )
        print(f"SUCCESS WandB initialized: {agent_name}_{env_type}_{model_name}")
        return wandb
    except Exception as e:
        print(f"WARNING WandB setup failed: {e}")
        return None


def log_to_wandb(wandb_instance, episode, result_dict, phase="training"):
    """Log metrics to WandB"""
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
        }

        # Add training-specific metrics
        if phase == "training":
            log_dict.update({
                "training/steps": result_dict.get('steps', 0),
                "training/fire_rate": result_dict.get('fire_rate', 0),
            })

        wandb_instance.log(log_dict)
    except Exception as e:
        print(f"WARNING WandB logging failed: {e}")


def log_validation_to_wandb(wandb_instance, episode, validation_results):
    """Log validation metrics to WandB"""
    if wandb_instance is None:
        return

    try:
        avg_reward = mean([r['reward'] for r in validation_results])
        success_rate = sum([r['success'] for r in validation_results]) / len(validation_results)
        fire_success_rate = sum([r['fire_success'] for r in validation_results]) / len(validation_results)
        avg_fire_attempts = mean([r['fire_attempts'] for r in validation_results])
        avg_locks = mean([r['locks'] for r in validation_results])

        wandb_instance.log({
            "validation/episode": episode,
            "validation/avg_reward": avg_reward,
            "validation/success_rate": success_rate,
            "validation/fire_success_rate": fire_success_rate,
            "validation/avg_fire_attempts": avg_fire_attempts,
            "validation/avg_locks": avg_locks,
            "validation/reward_std": pstdev([r['reward'] for r in validation_results]),
        })
    except Exception as e:
        print(f"WARNING WandB validation logging failed: {e}")


def validate(validationEpisodes, env: HarfangEnv, validationStep, agent, plot, plot_dir, arttir, model_dir, episode,
             checkpointRate, tensor_writer: SummaryWriter, highScore, successRate, if_random, wandb_instance=None):
    success = 0
    fire_success = 0
    valScores = []
    self_pos = []
    oppo_pos = []
    validation_results = []

    print(f"\n=== VALIDATION {arttir} STARTED ===")

    for e in range(validationEpisodes):
        distance = []
        fire = []
        lock = []
        missile = []
        if if_random:
            state = env.random_reset()
        else:
            state = env.reset()
        totalReward = 0
        done = False
        fire_attempts = 0
        successful_fires = 0
        lock_steps = 0

        for step in range(validationStep):
            if not done:
                if hasattr(agent, 'chooseActionNoNoise'):
                    action = agent.chooseActionNoNoise(state)
                else:  # SAC agent
                    action = agent.exploit(state)
                n_state, reward, done, info, iffire, beforeaction, afteraction, locked, step_success = env.step_test(
                    action)
                state = n_state
                totalReward += reward

                distance.append(env.loc_diff)
                if iffire:
                    fire.append(step)
                    fire_attempts += 1
                    if step_success == 1:
                        successful_fires += 1
                if locked:
                    lock.append(step)
                    lock_steps += 1
                if beforeaction:
                    missile.append(step)

                if e == validationEpisodes - 1:
                    self_pos.append(env.get_pos())
                    oppo_pos.append(env.get_oppo_pos())

                if step == validationStep - 1:
                    break

            elif done:
                if env.episode_success:
                    success += 1
                if env.fire_success:
                    fire_success += 1
                break

        valScores.append(totalReward)

        # Store validation episode result for WandB
        val_result = {
            'reward': totalReward,
            'fire_attempts': fire_attempts,
            'successful_fires': successful_fires,
            'locks': lock_steps,
            'success': env.episode_success,
            'fire_success': env.fire_success,
            'distance': distance[-1] if distance else 0
        }
        validation_results.append(val_result)

        if e < validationEpisodes:  # Log first 5 episodes in detail
            print(
                f"Val Episode {e + 1}: Reward={totalReward:.1f}, Fire attempts={fire_attempts}, Successful fires={successful_fires}, Lock steps={lock_steps}, Distance={distance[-1]:.1f}")

    avg_reward = mean(valScores)
    success_rate = success / validationEpisodes
    fire_success_rate = fire_success / validationEpisodes

    print(
        f"Validation Results: Avg Reward={avg_reward:.1f}, Success Rate={success_rate:.3f}, Fire Success Rate={fire_success_rate:.3f}")

    # Log validation results to WandB
    log_validation_to_wandb(wandb_instance, episode, validation_results)

    if avg_reward > highScore or success_rate >= successRate or arttir % 5 == 0:
        if hasattr(agent, 'saveCheckpoints'):
            agent.saveCheckpoints("Agent{}_{}_{}_".format(arttir, round(success_rate * 100), round(avg_reward)),
                                  model_dir)
        else:  # SAC agent
            agent.save_models("Agent{}_{}_{}_".format(arttir, round(success_rate * 100), round(avg_reward)))

        if plot:
            plot_3d_trajectories(self_pos, oppo_pos, fire, lock, plot_dir, f'trajectories_{arttir}.png')
            plot_distance(distance, lock, missile, fire, plot_dir, f'distance_{arttir}.png')

        if avg_reward > highScore:
            highScore = avg_reward
        if success_rate >= successRate:
            successRate = success_rate

    tensor_writer.add_scalar('Validation/Avg Reward', avg_reward, episode)
    tensor_writer.add_scalar('Validation/Std Reward', pstdev(valScores), episode)
    tensor_writer.add_scalar('Validation/Success Rate', success_rate, episode)
    tensor_writer.add_scalar('Validation/Fire Success Rate', fire_success_rate, episode)

    return highScore, successRate


def save_parameters_to_txt(log_dir, **kwargs):
    filename = os.path.join(log_dir, "log1.txt")
    with open(filename, 'w') as file:
        for key, value in kwargs.items():
            file.write(f"{key}={value}\n")


def main(config):
    print('GPU is available: ' + str(torch.cuda.is_available()))

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
    else:
        print("No seed is set")

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

    # ENVIRONMENT SETUP
    if env_type == "straight_line":
        print("Environment: Harfang straight line")
        trainingEpisodes = 2000
        validationEpisodes = 10  # Reduced for faster feedback
        explorationEpisodes = 200  # Reduced from 200
        maxStep = 3000
        validationStep = 3000
        env = HarfangEnv()

    elif env_type == "serpentine":
        print("Environment: Harfang serpentine")
        trainingEpisodes = 6000
        validationEpisodes = 20
        explorationEpisodes = 100
        maxStep = 1500
        validationStep = 1500
        env = HarfangSerpentineEnv()

    elif env_type == "circular":
        print("Environment: Harfang circular")
        trainingEpisodes = 6000
        validationEpisodes = 20
        explorationEpisodes = 100
        maxStep = 1900
        validationStep = 1900
        env = HarfangCircularEnv()

    df.set_renderless_mode(render)
    df.set_client_update_mode(True)

    # HYPERPARAMETERS - Adjusted for better learning
    bufferSize = 10 ** 5
    gamma = 0.99
    criticLR = 1e-3
    actorLR = 1e-3
    tau = 0.005
    checkpointRate = 50  # More frequent validation
    logRate = 100  # More frequent logging
    highScore = -math.inf
    successRate = 0
    batchSize = 128
    hiddenLayer1 = 256
    hiddenLayer2 = 512
    stateDim = 13
    actionDim = 4
    useLayerNorm = True

    # LOGGING SETUP
    start_time = datetime.datetime.now()
    log_dir = local_config["experiment"][
                  "result_dir"] + "/" + env_type + "/" + agent_name + "/" + model_name + "/" + str(
        start_time.year) + '_' + str(start_time.month) + '_' + str(start_time.day) + '_' + str(
        start_time.hour) + '_' + str(start_time.minute)
    model_dir = os.path.join(log_dir, 'model')
    summary_dir = os.path.join(log_dir, 'summary')
    plot_dir = os.path.join(log_dir, 'plot')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # AGENT SETUP
    if agent_name == 'TD3':
        print('Agent: TD3')
        agent = TD3Agent(actorLR, criticLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, tau, gamma, bufferSize,
                         batchSize, useLayerNorm, name)
    elif agent_name == 'SAC':
        print('Agent: SAC')
        state_space = gym.spaces.Box(low=np.array([-1.0] * stateDim), high=np.array([1.0] * stateDim), dtype=np.float64)
        action_space = gym.spaces.Box(low=np.array([-1.0] * actionDim), high=np.array([1.0] * actionDim),
                                      dtype=np.float64)
        agent = SACAgent(observation_space=state_space, action_space=action_space, log_dir=summary_dir,
                         batch_size=batchSize, lr=actorLR, hidden_units=[hiddenLayer1, hiddenLayer2],
                         memory_size=bufferSize, gamma=gamma, tau=tau)
    else:
        raise ValueError(f"Unknown agent type: {agent_name}. Please use 'TD3' or 'SAC'.")

    # SETUP WANDB
    wandb_config = {
        'trainingEpisodes': trainingEpisodes,
        'explorationEpisodes': explorationEpisodes,
        'maxStep': maxStep,
        'bufferSize': bufferSize,
        'batchSize': batchSize,
        'actorLR': actorLR,
        'criticLR': criticLR,
        'gamma': gamma,
        'tau': tau,
    }
    wandb_instance = setup_wandb(wandb_config, agent_name, env_type, model_name)

    save_parameters_to_txt(log_dir=log_dir, bufferSize=bufferSize, criticLR=criticLR, actorLR=actorLR,
                           batchSize=batchSize, maxStep=maxStep, validationStep=validationStep,
                           hiddenLayer1=hiddenLayer1, hiddenLayer2=hiddenLayer2, agent=agent_name, model_dir=model_dir)
    env.save_parameters_to_txt(log_dir)

    if agent_name == 'SAC':
        writer = agent.writer
    else:
        writer = SummaryWriter(summary_dir)

    arttir = 1
    if load_model:
        if agent_name == 'TD3':
            agent.loadCheckpoints("Agent20_64_100", model_dir)
        else:
            print("Loading SAC models - implement as needed")

    # RANDOM EXPLORATION
    print(f"\n=== EXPLORATION PHASE ({explorationEpisodes} episodes) ===")
    exploration_fire_attempts = 0
    exploration_successful_fires = 0
    exploration_locks = 0

    for episode in range(explorationEpisodes):
        if if_random:
            state = env.random_reset()
        else:
            state = env.reset()
        done = False
        episode_fire_attempts = 0
        episode_successful_fires = 0
        episode_locks = 0

        for step in range(maxStep):
            if not done:
                action = env.action_space.sample()
                n_state, reward, done, info, step_success = env.step(action)

                # Track exploration statistics
                if action[3] > 0:  # Fire attempt
                    episode_fire_attempts += 1
                    if step_success == 1:
                        episode_successful_fires += 1
                    elif step_success == -1:
                        pass  # Failed fire

                if n_state[7] > 0:  # Target locked
                    episode_locks += 1

                if step == maxStep - 1:
                    break

                if agent_name == 'TD3':
                    agent.store(state, action, n_state, reward, done, step_success)
                else:  # SAC
                    agent.memory.append(state, action, reward, n_state, done, done)

                state = n_state
            else:
                break

        exploration_fire_attempts += episode_fire_attempts
        exploration_successful_fires += episode_successful_fires
        exploration_locks += episode_locks

        # Log exploration to WandB
        exploration_result = {
            'reward': 0,  # No meaningful reward during exploration
            'fire_attempts': episode_fire_attempts,
            'successful_fires': episode_successful_fires,
            'locks': episode_locks,
            'success': False,
            'fire_success': False,
            'done': done,
            'steps': step + 1,
            'fire_rate': episode_successful_fires / max(episode_fire_attempts, 1),
            'min_distance': 0,
        }
        log_to_wandb(wandb_instance, episode, exploration_result, "exploration")

        if episode % 20 == 0:
            print(
                f"Exploration {episode + 1}/{explorationEpisodes}: Fire attempts={episode_fire_attempts}, Successful={episode_successful_fires}, Locks={episode_locks}")

    print(
        f"Exploration Summary: Total fires={exploration_fire_attempts}, Successful={exploration_successful_fires}, Total locks={exploration_locks}")
    print(
        f"Fire success rate during exploration: {exploration_successful_fires / max(exploration_fire_attempts, 1):.3f}")

    print(f"\n=== TRAINING PHASE ({trainingEpisodes} episodes) ===")
    scores = []
    trainsuccess = []
    firesuccess = []

    for episode in range(trainingEpisodes):
        if if_random:
            state = env.random_reset()
        else:
            state = env.reset()
        totalReward = 0
        done = False
        shut_down = False
        fire = False
        episode_fire_attempts = 0
        episode_successful_fires = 0
        episode_locks = 0
        min_distance = float('inf')
        # agent = sb3.PPO()
        agent =RuleAgent()
        for step in range(maxStep):
            if not done:
                # if agent_name == 'TD3':
                #     action = agent.chooseAction(state)
                # else:  # SAC
                #     action = agent.explore(state)
                action = agent.chooseAction(state)

                n_state, reward, done, info, step_success = env.step(action)

                # Track detailed statistics
                if action[3] > 0:  # Fire attempt
                    episode_fire_attempts += 1
                    if step_success == 1:
                        episode_successful_fires += 1

                if n_state[7] > 0:  # Target locked
                    episode_locks += 1

                min_distance = min(min_distance, env.loc_diff)

                if step == maxStep - 1:
                    break

                if agent_name == 'TD3':
                    agent.store(state, action, n_state, reward, done, step_success)
                else:  # SAC
                    agent.memory.append(state, action, reward, n_state, done, done)

                state = n_state
                totalReward += reward

                # LEARNING
                if agent_name == 'TD3':
                    if agent.buffer.fullEnough(agent.batchSize):
                        critic_loss, actor_loss = agent.learn()
                        if step % logRate == 0:
                            writer.add_scalar('Loss/Critic_Loss', critic_loss, step + episode * maxStep)
                            writer.add_scalar('Loss/Actor_Loss', actor_loss, step + episode * maxStep)

                            # Log losses to WandB
                            if wandb_instance:
                                wandb_instance.log({
                                    "training/critic_loss": critic_loss,
                                    "training/actor_loss": actor_loss,
                                    "training/step": step + episode * maxStep
                                })
                else:  # SAC
                    if len(agent.memory) > agent.batch_size:
                        agent.learn()

            elif done:
                if env.episode_success:
                    shut_down = True
                if env.fire_success:
                    fire = True
                break

        scores.append(totalReward)
        if shut_down:
            trainsuccess.append(1)
        else:
            trainsuccess.append(0)
        if fire:
            firesuccess.append(1)
        else:
            firesuccess.append(0)

        # Log training episode to WandB
        fire_rate = episode_successful_fires / max(episode_fire_attempts, 1)
        training_result = {
            'reward': totalReward,
            'min_distance': min_distance,
            'fire_attempts': episode_fire_attempts,
            'successful_fires': episode_successful_fires,
            'locks': episode_locks,
            'success': shut_down,
            'fire_success': fire,
            'done': done,
            'steps': step + 1,
            'fire_rate': fire_rate,
        }
        log_to_wandb(wandb_instance, episode, training_result, "training")

        # Detailed episode logging
        if episode % 10 == 0 or episode < 20:
            print(
                f"Episode {episode + 1}: Reward={totalReward:.1f}, MinDist={min_distance:.1f}, Fires={episode_fire_attempts}, Success={episode_successful_fires}, Rate={fire_rate:.3f}, Locks={episode_locks}, Done={done}, Victory={shut_down}")

        writer.add_scalar('Training/Episode Reward', totalReward, episode)
        writer.add_scalar('Training/Last 100 Episode Average Reward', np.mean(scores[-100:]), episode)
        writer.add_scalar('Training/Average Step Reward', totalReward / step if step > 0 else 0, episode)
        writer.add_scalar('Training/Last 50 Episode Train success rate', np.mean(trainsuccess[-50:]), episode)
        writer.add_scalar('Training/Last 50 Episode Fire success rate', np.mean(firesuccess[-50:]), episode)
        writer.add_scalar('Training/Episode Fire Attempts', episode_fire_attempts, episode)
        writer.add_scalar('Training/Episode Successful Fires', episode_successful_fires, episode)
        writer.add_scalar('Training/Episode Locks', episode_locks, episode)
        writer.add_scalar('Training/Min Distance', min_distance, episode)

        # Log aggregated metrics to WandB
        if wandb_instance:
            wandb_instance.log({
                "training/avg_reward_last_100": np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores),
                "training/success_rate_last_50": np.mean(trainsuccess[-50:]) if len(trainsuccess) >= 50 else np.mean(
                    trainsuccess),
                "training/fire_success_rate_last_50": np.mean(firesuccess[-50:]) if len(firesuccess) >= 50 else np.mean(
                    firesuccess),
            })

        now = time.time()
        seconds = int((now - start) % 60)
        minutes = int(((now - start) // 60) % 60)
        hours = int((now - start) // 3600)

        # VALIDATION
        if (((episode + 1) % checkpointRate) == 0):
            print(f"\nRunning validation after episode {episode + 1}...")
            highScore, successRate = validate(validationEpisodes, env, validationStep, agent, plot, plot_dir, arttir,
                                              model_dir, episode, checkpointRate, writer, highScore, successRate,
                                              if_random, wandb_instance)
            arttir += 1
            print(f"Continuing training... ({hours:02d}:{minutes:02d}:{seconds:02d} elapsed)")

    # Finish WandB run
    if wandb_instance:
        # Log final summary metrics
        final_success_rate = np.mean(trainsuccess)
        final_avg_reward = np.mean(scores)
        final_fire_success_rate = np.mean(firesuccess)

        wandb_instance.log({
            "final/success_rate": final_success_rate,
            "final/avg_reward": final_avg_reward,
            "final/fire_success_rate": final_fire_success_rate,
            "final/total_episodes": trainingEpisodes,
        })

        wandb_instance.finish()
        print("SUCCESS WandB run completed and finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='TD3', choices=['TD3', 'SAC'],
                        help="Specify the agent to use: 'TD3' or 'SAC'. Default is 'TD3'.")
    parser.add_argument('--port', type=int, default=None,
                        help="The port number for the training environment. Example: 12345.")
    parser.add_argument('--model_name', type=str, default=None,
                        help="Name of the model to be saved. Example: 'TD3_experiment1'.")
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