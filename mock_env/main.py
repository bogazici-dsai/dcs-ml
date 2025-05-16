from combat_mission_env import CombatMissionEnv
import numpy as np
import pygame
from pygame.locals import *
import sys
import time

def handle_input(env):
    """Handle keyboard input for interactive mode"""
    # Ensure pygame is initialized
    if not pygame.get_init():
        pygame.init()
        if env.screen is None:
            env.screen = pygame.display.set_mode(env.screen_size)
            pygame.display.set_caption("Enhanced Combat Mission Environment")

    # Default to no-op if no key was pressed
    action = 11  # No-op

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == KEYDOWN:
            # Weapon controls
            if event.key == K_1:
                return 0  # Fire Active Radar Missile
            if event.key == K_2:
                return 1  # Fire IR Guided Missile

            # Sensor controls
            if event.key == K_r:
                return 2  # Toggle Radar
            if event.key == K_s:
                return 3  # Toggle Radar Sweep Mode

            # Autopilot
            if event.key == K_a:
                return 4  # Toggle Autopilot

            # Movement controls (only work with autopilot off)
            if event.key == K_t:
                return 5  # Toward enemy
            if event.key == K_UP:
                return 6  # North
            if event.key == K_DOWN:
                return 7  # South
            if event.key == K_RIGHT:
                return 8  # East
            if event.key == K_LEFT:
                return 9  # West
            if event.key == K_f:
                return 10  # Away from enemy (flee)

            # No operation
            if event.key == K_SPACE:
                return 11  # No-op

            # Quit
            if event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()

    # Use key states for continuous key-hold movement
    keys = pygame.key.get_pressed()
    if not env.autopilot_on:  # Only process movement if autopilot is off
        if keys[K_UP]:
            return 6  # North
        if keys[K_DOWN]:
            return 7  # South
        if keys[K_RIGHT]:
            return 8  # East
        if keys[K_LEFT]:
            return 9  # West
        if keys[K_t]:
            return 5  # Toward enemy
        if keys[K_f]:
            return 10  # Away from enemy

    return action


def display_controls():
    """Display the control scheme for interactive mode"""
    controls = [
        "CONTROLS:",
        "1: Fire Active Radar Missile",
        "2: Fire IR Guided Missile",
        "r: Toggle Radar On/Off",
        "s: Toggle Radar Sweep Mode",
        "a: Toggle Autopilot",
        "t: Move Toward Enemy",
        "Arrow Keys: Move North/South/East/West",
        "f: Move Away from Enemy",
        "Space: No Operation",
        "ESC: Quit"
    ]

    for i, line in enumerate(controls):
        print(line)


def run_simulation(steps=300, interactive=False):
    env = CombatMissionEnv()
    obs = env.reset()

    total_reward = 0
    action_names = {
        0: "Fire Active Radar", 1: "Fire IR Guided", 2: "Toggle Radar",
        3: "Toggle Sweep Mode", 4: "Toggle Autopilot", 5: "Toward Enemy",
        6: "North", 7: "South", 8: "East", 9: "West",
        10: "Away from Enemy", 11: "No-op"
    }

    # Initialize pygame for interactive mode
    if interactive:
        if not pygame.get_init():
            pygame.init()
        display_controls()

    # Render first frame
    env.render()

    try:
        for step in range(steps):
            if interactive:
                # Handle keyboard input for interactive mode
                action = handle_input(env)
            else:
                # Random action for non-interactive mode
                action = env.action_space.sample()

            obs, reward, done, info = env.step(action)
            total_reward += reward

            env.render()

            # Display action and reward information
            print(f"Step {step + 1}: Action: {action_names[action]}, Reward: {reward:.2f}, "
                  f"Total Reward: {total_reward:.2f}")

            # Add a small delay for visibility
            pygame.time.delay(50)  # 50ms delay

            if done:
                print(f"Episode finished after {step + 1} steps with total reward: {total_reward:.2f}")
                print("Resetting environment...")
                pygame.time.delay(1000)  # 1 second delay
                obs = env.reset()
                total_reward = 0
    finally:
        # Ensure environment is closed properly
        env.close()(f"Episode finished after {step + 1} steps with total reward: {total_reward:.2f}")
        print("Resetting environment...")
        time.sleep(1)
        obs = env.reset()
        total_reward = 0





def run_ai_agent():
    """Run simulation with the rule-based AI agent"""
    from rule_agent import run_rule_based_simulation
    run_rule_based_simulation()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Combat Mission Simulator')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['random', 'interactive', 'ai'],
                        help='Simulation mode: random actions, interactive control, or AI agent')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of steps to run the simulation')

    args = parser.parse_args()

    if args.mode == 'interactive':
        run_simulation(steps=args.steps, interactive=True)
    elif args.mode == 'ai':
        run_ai_agent()
    else:  # random
        run_simulation(steps=args.steps, interactive=False)
