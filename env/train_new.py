import os
import argparse
import time

import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from hirl.environments.HarfangEnv_GYM_new import HarfangEnv  # SimpleEnemy kaldırıldı
import hirl.environments.dogfight_client as df

from agents import Ally, Oppo
from action_helper import *

import wandb

WANDB = False
WANDB_RUN_NAME = "train_harfang_rule_based_test_04"

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
    env = HarfangEnv()

    # Agents
    ally = Ally(steps_between_fires=args.fire_cooldown, debug=args.debug)
    oppo = Oppo(debug=args.debug, fire_cooldown=args.fire_cooldown)

    scores, successes, evade_successes = [], [], []
    os.makedirs("trajectories", exist_ok=True)

    if __name__ == "__main__":
        if WANDB:
            wandb.init(
                project="Harfang_Training",
                entity="BILGEM_DCS_RL",
                name=WANDB_RUN_NAME,
                config={
                    "env_name": "HARFANG",
                    "algo": "Rule-Based",
                    "max_steps": 10000
                },
                sync_tensorboard=True
            )


    for ep in range(args.episodes):
        # reset() returns (ally_obs_dict, oppo_obs_dict)
        state, oppo_state = env.reset()

        ally.update(state)
        oppo.update(oppo_state)

        done = False
        total_reward, steps = 0.0, 0
        max_steps = 10000

        # Optional trajectory plotting
        agent_positions, oppo_positions = [], []
        rewards, dones = [], []

        while steps < max_steps and not done:
            # Rule-based actions
            action, ally_cmd = ally.behave()
            oppo_action, oppo_cmd = oppo.behave()

            # Step (gym-style 4-tuple)
            n_state, reward, done, info = env.step(action, oppo_action)

            # Carry next states
            state = n_state
            oppo_state = info.get("opponent_obs", oppo_state)

            ally.update(state)
            oppo.update(oppo_state)

            ally_health = state.get("ally_health")
            oppo_health = state.get("oppo_health")
            if steps % 1000 == 0:
                print(f"Ally Health: {ally_health:.1f} | Oppo Health {oppo_health:.1f}")
            total_reward += float(reward)
            rewards.append(float(reward))
            dones.append(bool(done))
            steps += 1

            # record 3D positions (meters)
            # env dict mapping: plane_x, plane_y, plane_z are normalized
            agent_positions.append([
                state.get("plane_x", 0.0) * 10000.0,
                state.get("plane_y", 0.0) * 10000.0,
                state.get("plane_z", 0.0) * 10000.0
            ])
            oppo_positions.append([
                oppo_state.get("plane_x", 0.0) * 10000.0,
                oppo_state.get("plane_y", 0.0) * 10000.0,
                oppo_state.get("plane_z", 0.0) * 10000.0
            ])

        # Episode results

        evade_success = True if state.get("ally_health") >= 0.7 else False
        success = int(info.get("episode_success", False))
        scores.append(total_reward)
        successes.append(success)
        evade_successes.append(evade_success)
        ally_health = state.get("ally_health")
        oppo_health = state.get("oppo_health")
        print(f"Episode {ep + 1}/{args.episodes} | Reward: {total_reward:.1f} | "
              f"Success: {success} | Steps: {steps} | Altitude: {state.get('altitude',0.0)*10000:.0f} m | Ally Health: {ally_health:.1f} | Oppo Health {oppo_health:.1f}")

        # --- WandB logging ---
        if WANDB:
            wandb.log({
                "episode": ep + 1,
                "reward": total_reward,
                "success": success,
                "evade_success": int(evade_success),
                "steps": steps,
                "ally_health": ally_health,
                "oppo_health": oppo_health,
                "average_reward" : float(np.mean(scores)),
                "success_rate" : float(np.mean(successes)),
                "evade_success_rate" : float(np.mean(evade_successes)),
            })


        # Optional 3D plot
        if args.plot:

            agent_positions = np.array(agent_positions, dtype=float)
            oppo_positions = np.array(oppo_positions, dtype=float)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(agent_positions[:, 0], agent_positions[:, 1], agent_positions[:, 2], label="Agent")
            ax.plot(oppo_positions[:, 0], oppo_positions[:, 1], oppo_positions[:, 2], label="Opponent")
            ax.scatter(agent_positions[0, 0], agent_positions[0, 1], agent_positions[0, 2],
                       marker='o', label='Agent Start')
            ax.scatter(oppo_positions[0, 0], oppo_positions[0, 1], oppo_positions[0, 2],
                       marker='o', label='Opponent Start')
            ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
            ax.legend()
            ax.set_title(f"3D Trajectory (Episode {ep+1})")
            plt.tight_layout()
            plt.savefig(f"trajectories/episode_{ep+1:03d}.png")
            plt.close()
            print(f"Trajectory saved to trajectories/episode_{ep+1:03d}.png")

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
    parser.add_argument('--env', type=str, default='harfang', choices=['harfang'])
    parser.add_argument('--port', type=int, default=50888)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--plot', action='store_true', help="Save 3D trajectory plots per episode.")
    parser.add_argument('--debug', action='store_true', help="Verbose rule-based decisions.")
    parser.add_argument('--fire_cooldown', type=int, default=600,
                        help="Min steps between ally fires.")
    args = parser.parse_args()
    main(args)
