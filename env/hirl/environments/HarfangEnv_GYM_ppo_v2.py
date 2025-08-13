# HarfangEnv_GYM_ppo.py
import numpy as np
import gym
import random
import os
import inspect
import hirl.environments.dogfight_client as df
from hirl.environments.constants import *


class HarfangEnv(gym.Env):
    def __init__(self, max_episode_steps=None):
        super().__init__()
        self.done = False
        self.loc_diff = 0
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        self.Plane_ID_oppo = "ennemy_2"
        self.Plane_ID_ally = "ally_1"
        self.Aircraft_Loc = None
        self.Ally_target_locked = False
        self.n_Ally_target_locked = False
        self.reward = 0
        self.Plane_Irtifa = 0
        self.now_missile_state = False
        self.missile1_state = True
        self.n_missile1_state = True
        self.missile = df.get_machine_missiles_list(self.Plane_ID_ally)
        self.missile1_id = self.missile[0] if len(self.missile) > 0 else None
        self.oppo_health = 0.2
        self.target_angle = None
        self.success = 0
        self.episode_success = False
        self.fire_success = False

        # MAX STEP LIMIT - YENİ EKLENEN
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Reward tracking variables - ENHANCED
        self.lock_duration = 0
        self.prev_action = [0, 0, 0, 0]
        self.shots_fired = 0
        self.consecutive_locks = 0
        self.prev_distance = None
        self.altitude_violations = 0
        self.reward_components = {}  # Track individual reward components

    def reset(self):
        self.Ally_target_locked = False
        self.n_Ally_target_locked = False
        self.missile1_state = True
        self.n_missile1_state = True
        self.success = 0
        self.done = False
        self._reset_machine()
        self._reset_missile()
        state_ally = self._get_observation()
        df.set_target_id(self.Plane_ID_ally, self.Plane_ID_oppo)
        self.episode_success = False
        self.fire_success = False
        self.state = state_ally

        # RESET STEP COUNTER AND REWARD VARIABLES - ENHANCED
        self.current_step = 0
        self.lock_duration = 0
        self.prev_action = [0, 0, 0, 0]
        self.shots_fired = 0
        self.consecutive_locks = 0
        self.prev_distance = None
        self.altitude_violations = 0
        self.reward_components = {}

        return np.array(state_ally, dtype=np.float32)

    def random_reset(self):
        self.Ally_target_locked = False
        self.n_Ally_target_locked = False
        self.missile1_state = True
        self.n_missile1_state = True
        self.success = 0
        self.done = False
        self._random_reset_machine()
        self._reset_missile()
        state_ally = self._get_observation()
        df.set_target_id(self.Plane_ID_ally, self.Plane_ID_oppo)
        self.episode_success = False
        self.fire_success = False
        self.state = state_ally

        # RESET STEP COUNTER AND REWARD VARIABLES - ENHANCED
        self.current_step = 0
        self.lock_duration = 0
        self.prev_action = [0, 0, 0, 0]
        self.shots_fired = 0
        self.consecutive_locks = 0
        self.prev_distance = None
        self.altitude_violations = 0
        self.reward_components = {}

        return np.array(state_ally, dtype=np.float32)

    def _random_reset_machine(self):
        df.reset_machine("ally_1")
        df.reset_machine("ennemy_2")
        df.set_health("ennemy_2", 0.2)
        self.oppo_health = 0.2
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.reset_machine_matrix(
            self.Plane_ID_ally,
            0 + random.randint(-100, 100),
            3500 + random.randint(-100, 100),
            -4000 + random.randint(-100, 100), 0, 0, 0
        )
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def step(self, action):
        # INCREMENT STEP COUNTER
        self.current_step += 1

        self._apply_action(action)
        n_state = self._get_observation()
        self._get_reward(self.state, action, n_state)
        self.state = n_state
        self._get_termination()

        # CHECK MAX STEP LIMIT
        if self.max_episode_steps is not None and self.current_step >= self.max_episode_steps:
            self.done = True
            # print(f"Episode terminated: Max steps ({self.max_episode_steps}) reached")

        # Enhanced info dictionary for better debugging and logging
        info = {
            'success': int(self.episode_success),
            'fire_success': int(self.fire_success),
            'step_success': self.success,
            'distance': self.loc_diff,
            'altitude': self.Plane_Irtifa,
            'target_locked': int(self.n_Ally_target_locked),
            'reward_components': self.reward_components.copy()
        }

        return np.array(n_state, dtype=np.float32), float(self.reward), bool(self.done), info

    def _get_reward(self, state, action, n_state):
        """IMPROVED REWARD FUNCTION WITH BETTER SCALING AND BALANCE"""
        self.reward = 0.0
        self.success = 0
        self.reward_components = {}  # Track components for debugging
        self._get_loc_diff()  # Calculate current distance to enemy

        # 1. Distance shaping (encourage to approach the opponent) - IMPROVED
        dx, dy, dz = n_state[0], n_state[1], n_state[2]
        prev_dx, prev_dy, prev_dz = state[0], state[1], state[2]
        prev_distance = np.sqrt(prev_dx ** 2 + prev_dy ** 2 + prev_dz ** 2)
        curr_distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        distance_change = prev_distance - curr_distance

        # More balanced distance reward - scaled to prevent dominance
        distance_reward = 1.5 * np.clip(distance_change, -1, 1)
        self.reward += distance_reward
        self.reward_components['distance_change'] = distance_reward

        # 2. Direct distance penalty (discourage staying far) - IMPROVED SCALING
        distance_penalty = -0.001 * min(self.loc_diff, 5000)  # Cap penalty at reasonable distance
        self.reward += distance_penalty
        self.reward_components['distance_penalty'] = distance_penalty

        # 3. Target angle reward (reward looking at the enemy) - IMPROVED
        target_angle_reward = 1.5 * (1.0 - self.target_angle)  # Slightly reduced
        self.reward += target_angle_reward
        self.reward_components['target_angle'] = target_angle_reward

        # 4. Lock-on bonus (encourage maintaining lock) - NEW
        if self.n_Ally_target_locked:
            self.consecutive_locks += 1
            lock_bonus = min(0.5 * self.consecutive_locks, 2.0)  # Progressive bonus, capped
            self.reward += lock_bonus
            self.reward_components['lock_bonus'] = lock_bonus
        else:
            self.consecutive_locks = 0
            self.reward_components['lock_bonus'] = 0.0

        # 5. Altitude zone shaping - IMPROVED GRADUAL PENALTIES
        altitude_reward = 0.0
        if self.Plane_Irtifa < 2000:
            altitude_reward = -5.0 * (2000 - self.Plane_Irtifa) / 1000  # Gradual penalty
        elif self.Plane_Irtifa > 7000:
            altitude_reward = -5.0 * (self.Plane_Irtifa - 7000) / 1000  # Gradual penalty

        if self.Plane_Irtifa < 600:  # Dangerous low altitude
            altitude_reward = -50.0
            self.altitude_violations += 1
        elif self.Plane_Irtifa > 10000:  # Dangerous high altitude
            altitude_reward = -50.0
            self.altitude_violations += 1

        self.reward += altitude_reward
        self.reward_components['altitude'] = altitude_reward

        # 6. Missile firing logic - IMPROVED FEEDBACK
        firing_reward = 0.0
        if self.now_missile_state:
            if self.missile1_state and not self.Ally_target_locked:
                firing_reward = -5.0  # Reduced penalty for firing without lock
                self.success = -1
                # print('Failed to fire: no lock')
            elif self.missile1_state and self.Ally_target_locked:
                firing_reward = 15.0  # Increased reward for good fire
                # print('Successful fire!')
                self.success = 1
                self.fire_success = True
            else:
                firing_reward = -1.0  # Mild penalty for random shots

        self.reward += firing_reward
        self.reward_components['firing'] = firing_reward

        # 7. Enemy defeated (terminal reward) - BETTER SCALED
        victory_reward = 0.0
        if self.oppo_health['health_level'] <= 0.1 and self.fire_success:
            victory_reward = 100.0  # Reduced from 250 to prevent dominance
            # print('Enemy destroyed!')

        self.reward += victory_reward
        self.reward_components['victory'] = victory_reward

        # 8. Time penalty (small per step to encourage efficiency) - REDUCED
        time_penalty = -0.02  # Reduced from -0.05
        self.reward += time_penalty
        self.reward_components['time_penalty'] = time_penalty

        # 9. Action smoothness reward (discourage erratic movements) - NEW
        if hasattr(self, 'prev_action'):
            action_change = np.sum(np.abs(np.array(action[:3]) - np.array(self.prev_action[:3])))
            smoothness_reward = -0.1 * action_change  # Small penalty for large action changes
            self.reward += smoothness_reward
            self.reward_components['smoothness'] = smoothness_reward
        else:
            self.reward_components['smoothness'] = 0.0

        self.prev_action = action.copy()

        # 10. Engagement zone reward (encourage staying in optimal combat range) - NEW
        engagement_reward = 0.0
        if 1000 <= self.loc_diff <= 3000:  # Optimal engagement range
            engagement_reward = 0.5
        elif 500 <= self.loc_diff <= 5000:  # Acceptable range
            engagement_reward = 0.2
        else:  # Too close or too far
            engagement_reward = -0.3

        self.reward += engagement_reward
        self.reward_components['engagement_range'] = engagement_reward

        # Store total reward for debugging
        self.reward_components['total'] = self.reward

        return self.reward

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))
        df.set_plane_pitch(self.Plane_ID_oppo, float(0))
        df.set_plane_roll(self.Plane_ID_oppo, float(0))
        df.set_plane_yaw(self.Plane_ID_oppo, float(0))

        if float(action_ally[3] > 0):
            df.fire_missile(self.Plane_ID_ally, 0)
            self.now_missile_state = True
        else:
            self.now_missile_state = False

        df.update_scene()

    def _get_termination(self):
        # Altitude limits - MORE FORGIVING
        if self.Plane_Irtifa < 400 or self.Plane_Irtifa > 12000:  # Extended safe range
            self.done = True
            # print(f"Episode terminated: Altitude limit (altitude: {self.Plane_Irtifa:.1f}m)")

        # Enemy health
        if self.oppo_health['health_level'] <= 0:
            self.done = True
            self.episode_success = True
            # print(f"Episode terminated: Enemy destroyed! VICTORY")

        # Safety termination for excessive altitude violations
        if self.altitude_violations > 50:  # Allow some violations before terminating
            self.done = True
            # print(f"Episode terminated: Too many altitude violations")

    def _reset_machine(self):
        df.reset_machine("ally_1")
        df.reset_machine("ennemy_2")
        df.set_health("ennemy_2", 0.2)
        self.oppo_health = 0.2
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def _reset_missile(self):
        self.now_missile_state = False
        df.rearm_machine(self.Plane_ID_ally)

    def _get_loc_diff(self):
        self.loc_diff = (((self.Aircraft_Loc[0] - self.Oppo_Loc[0]) ** 2) +
                         ((self.Aircraft_Loc[1] - self.Oppo_Loc[1]) ** 2) +
                         ((self.Aircraft_Loc[2] - self.Oppo_Loc[2]) ** 2)) ** 0.5

    def _get_observation(self):
        Plane_state = df.get_plane_state(self.Plane_ID_ally)
        Plane_Pos = [Plane_state["position"][0] / NormStates["Plane_position"],
                     Plane_state["position"][1] / NormStates["Plane_position"],
                     Plane_state["position"][2] / NormStates["Plane_position"]]
        Plane_Euler = [Plane_state["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
                       Plane_state["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                       Plane_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]
        self.Plane_Irtifa = Plane_state["position"][1]
        self.Aircraft_Loc = Plane_state["position"]
        Oppo_state = df.get_plane_state(self.Plane_ID_oppo)
        Oppo_Pos = [Oppo_state["position"][0] / NormStates["Plane_position"],
                    Oppo_state["position"][1] / NormStates["Plane_position"],
                    Oppo_state["position"][2] / NormStates["Plane_position"]]
        Oppo_Euler = [Oppo_state["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
                      Oppo_state["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                      Oppo_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]
        self.Oppo_Loc = Oppo_state["position"]
        self.Ally_target_locked = self.n_Ally_target_locked
        self.n_Ally_target_locked = Plane_state["target_locked"]
        locked = 1 if self.n_Ally_target_locked else -1
        target_angle = Plane_state['target_angle'] / 180
        self.target_angle = target_angle
        Pos_Diff = [Plane_Pos[0] - Oppo_Pos[0], Plane_Pos[1] - Oppo_Pos[1], Plane_Pos[2] - Oppo_Pos[2]]
        self.oppo_health = df.get_health(self.Plane_ID_oppo)
        oppo_hea = self.oppo_health['health_level']
        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_ally)
        self.missile1_state = self.n_missile1_state
        self.n_missile1_state = Missile_state["missiles_slots"][0] if Missile_state["missiles_slots"] else False
        missile1_state = 1 if self.n_missile1_state else -1
        States = np.concatenate(
            (Pos_Diff, Plane_Euler, [target_angle], [locked], [missile1_state], Oppo_Euler, [oppo_hea]), axis=None)
        return States.astype(np.float32)

    def get_pos(self):
        plane_state = df.get_plane_state(self.Plane_ID_ally)
        return np.array([plane_state["position"][0], plane_state["position"][1], plane_state["position"][2]])

    def get_oppo_pos(self):
        plane_state = df.get_plane_state(self.Plane_ID_oppo)
        return np.array([plane_state["position"][0], plane_state["position"][1], plane_state["position"][2]])

    def save_parameters_to_txt(self, log_dir):
        source_code1 = inspect.getsource(self._get_reward)
        source_code2 = inspect.getsource(self._reset_machine)
        source_code3 = inspect.getsource(self._get_termination)
        filename = os.path.join(log_dir, "log2.txt")
        with open(filename, 'w', encoding='utf-8') as file:
            file.write("IMPROVED REWARD FUNCTION WITH BALANCED COMPONENTS\n")
            file.write("=" * 50 + "\n")
            file.write(source_code1)
            file.write('\n' + "=" * 50 + "\n")
            file.write(source_code2)
            file.write('\n' + "=" * 50 + "\n")
            file.write(source_code3)


# ----------- Child classes (with their subclass-specific variable inits) -----------

class HarfangSerpentineEnv(HarfangEnv):
    def __init__(self, max_episode_steps=None):
        super().__init__(max_episode_steps=max_episode_steps)
        self.serpentine_step = 0
        self.duration = 250
        self.oppo_yaw = -0.1

    def set_ennemy_yaw(self):
        self.serpentine_step += 1
        if self.serpentine_step % self.duration == 0:
            self.serpentine_step = 0
            self.oppo_yaw = 0.1 * (-1 if self.oppo_yaw > 0 else 1)
            self.duration = 500
        df.set_plane_pitch(self.Plane_ID_oppo, float(0))
        df.set_plane_roll(self.Plane_ID_oppo, float(0))
        df.set_plane_yaw(self.Plane_ID_oppo, float(self.oppo_yaw))

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))
        self.set_ennemy_yaw()
        if float(action_ally[3] > 0):
            df.fire_missile(self.Plane_ID_ally, 0)
            self.now_missile_state = True
        else:
            self.now_missile_state = False
        df.update_scene()

    def _reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = 0.2
        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def _random_reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = 0.2
        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(
            self.Plane_ID_ally,
            0 + random.randint(-100, 100),
            3500 + random.randint(-100, 100),
            -4000 + random.randint(-100, 100), 0, 0, 0
        )
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def reset_ennemy(self):
        self.oppo_yaw = -0.1
        self.serpentine_step = 0
        self.duration = 250
        df.reset_machine(self.Plane_ID_oppo)
        df.set_health(self.Plane_ID_oppo, 0.2)
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_oppo)


class HarfangCircularEnv(HarfangEnv):
    def __init__(self, max_episode_steps=None):
        super().__init__(max_episode_steps=max_episode_steps)
        self.circular_step = 0

    def set_ennemy_yaw(self):
        self.circular_step += 1
        if self.circular_step < 100:
            df.set_plane_pitch(self.Plane_ID_oppo, float(-0.02))
            df.set_plane_roll(self.Plane_ID_oppo, float(0.84))
        else:
            df.set_plane_pitch(self.Plane_ID_oppo, float(-0.01))
        df.set_plane_roll(self.Plane_ID_oppo, float(0.28))
        df.set_plane_yaw(self.Plane_ID_oppo, float(0))

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))
        self.set_ennemy_yaw()
        if float(action_ally[3] > 0):
            df.fire_missile(self.Plane_ID_ally, 0)
            self.now_missile_state = True
        else:
            self.now_missile_state = False
        df.update_scene()

    def _reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = 0.2
        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def _random_reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = 0.2
        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(
            self.Plane_ID_ally,
            0 + random.randint(-100, 100),
            3500 + random.randint(-100, 100),
            -4000 + random.randint(-100, 100), 0, 0, 0
        )
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def reset_ennemy(self):
        self.circular_step = 0
        df.reset_machine(self.Plane_ID_oppo)
        df.set_health(self.Plane_ID_oppo, 0.2)
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.8)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 290)
        df.retract_gear(self.Plane_ID_oppo)