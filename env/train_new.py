import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from hirl.environments.HarfangEnv_GYM_new import HarfangEnv, SimpleEnemy
import hirl.environments.dogfight_client as df

from agents import Ally, Oppo
from action_helper import *


# ------------------------------------ Main ------------------------------------ #
def main(args):
    # Network bootstrap
    with open('local_config.yaml', 'r') as f:
        local_config = yaml.safe_load(f)
    if local_config['network']['ip'] == 'YOUR_IP_ADDRESS':
        raise ValueError("Please update 'network.ip' in env/local_config.yaml")

    df.connect(local_config["network"]["ip"], args.port)
    df.disable_log()
    df.set_renderless_mode(not args.render)
    df.set_client_update_mode(True)

    # Env selection
    env = SimpleEnemy() if args.env == "simple_enemy" else HarfangEnv()

    # Agents
    ally = Ally(steps_between_fires=args.fire_cooldown, debug=args.debug)
    oppo = Oppo(debug=args.debug, fire_cooldown=args.fire_cooldown)

    scores, successes, evade_successes = [], [], []
    os.makedirs("trajectories", exist_ok=True)

    for ep in range(args.episodes):
        # NOTE: reset() returns (ally_obs, oppo_obs) for this script's usage.
        state, oppo_state = env.reset()

        ally.update(state)
        oppo.update(oppo_state)

        done = False
        total_reward, steps = 0.0, 0
        max_steps = 5000

        # Optional trajectory plotting
        agent_positions, oppo_positions = [], []
        rewards, dones = [], []

        while steps < max_steps and not done:
            # Rule-based actions
            action, ally_cmd = ally.behave()
            oppo_action, oppo_cmd = oppo.behave()

            # Step (gym-style 4-tuple)
            n_state, reward, done, info = env.step(action, oppo_action)

            # Carry next states for both sides
            state = n_state
            oppo_state = info.get("opponent_obs", oppo_state)

            ally.update(state)
            oppo.update(oppo_state)

            total_reward += float(reward)
            rewards.append(float(reward))
            dones.append(bool(done))
            steps += 1

            # record 3D positions (meters)
            agent_positions.append(state[13:16] * 10000.0)
            oppo_positions.append(oppo_state[13:16] * 10000.0)

        # Episode results
        evade_success = (state[20] >= 1.0)  # preserved check
        success = int(info.get("episode_success", False))
        scores.append(total_reward)
        successes.append(success)
        evade_successes.append(evade_success)

        # Optional logging hook
        # (left as comment; implement your own file logger if needed)
        # if hasattr(ally, "log_episode"): ally.log_episode(rewards, dones)

        print(f"Episode {ep + 1}/{args.episodes} | Reward: {total_reward:.1f} | Success: {success} | Steps: {steps} | Ally Health: {state[20]:.2f}")

        # Optional 3D plot
        if args.plot:
            agent_positions = np.array(agent_positions, dtype=float)
            oppo_positions = np.array(oppo_positions, dtype=float)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(agent_positions[:, 0], agent_positions[:, 2], agent_positions[:, 1], label="Agent")
            ax.plot(oppo_positions[:, 0], oppo_positions[:, 2], oppo_positions[:, 1], label="Opponent")
            ax.scatter(agent_positions[0, 0], agent_positions[0, 2], agent_positions[0, 1], marker='o', label='Agent Start')
            ax.scatter(oppo_positions[0, 0], oppo_positions[0, 2], oppo_positions[0, 1], marker='o', label='Opponent Start')
            ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
            ax.legend()
            ax.set_title(f"3D Trajectory (Episode {ep+1})")
            plt.tight_layout()
            plt.savefig(f"trajectories/episode_{ep+1:03d}.png")
            plt.close()

    # Summary
    avg_reward = float(np.mean(scores)) if scores else 0.0
    success_rate = float(np.mean(successes)) if successes else 0.0
    evade_success_rate = float(np.mean(evade_successes)) if evade_successes else 0.0
    print("\n=== Rule-Based Agent Results ===")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Evade Success Rate: {evade_success_rate:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='simple_enemy', choices=['simple_enemy', 'harfang'])
    parser.add_argument('--port', type=int, default=50888)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--plot', action='store_true', help="Save 3D trajectory plots per episode.")
    parser.add_argument('--debug', action='store_true', help="Verbose rule-based decisions.")
    parser.add_argument('--fire_cooldown', type=int, default=600, help="Min steps between ally fires.")
    parser.add_argument(
        '--agent',
        type=str,
        default='rule',
        choices=['rule', 'agents'],
        help="Select which agent type to use: 'rule' for Ally/Oppo rule-based, 'agents' for legacy Agents class."
    )
    args = parser.parse_args()
    main(args)