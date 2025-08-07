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

        # Reward tracking variables
        self.lock_duration = 0
        self.prev_action = [0, 0, 0, 0]
        self.shots_fired = 0

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

        # RESET STEP COUNTER AND REWARD VARIABLES
        self.current_step = 0
        self.lock_duration = 0
        self.prev_action = [0, 0, 0, 0]
        self.shots_fired = 0

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

        # RESET STEP COUNTER AND REWARD VARIABLES
        self.current_step = 0
        self.lock_duration = 0
        self.prev_action = [0, 0, 0, 0]
        self.shots_fired = 0

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

        return np.array(n_state, dtype=np.float32), float(self.reward), bool(self.done), {}

    def _get_reward(self, state, action, n_state):
        self.reward = 0.0
        self.success = 0
        self._get_loc_diff()  # Calculate current distance to enemy

        # 1. Distance shaping (encourage to approach the opponent)
        # Reward for getting closer (change in distance)
        dx, dy, dz = n_state[0], n_state[1], n_state[2]
        prev_dx, prev_dy, prev_dz = state[0], state[1], state[2]
        prev_distance = np.sqrt(prev_dx ** 2 + prev_dy ** 2 + prev_dz ** 2)
        curr_distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        distance_change = prev_distance - curr_distance
        self.reward += 2.0 * np.clip(distance_change, -1, 1)  # Moderate, symmetric

        # 2. Direct distance penalty (discourage staying far)
        self.reward -= 0.0015 * self.loc_diff  # Gentler than before

        # 3. Target angle penalty (reward looking at the enemy)
        self.reward += 2.0 * (1.0 - self.target_angle)  # 0 if facing away, +2 if aligned

        # 4. Altitude zone shaping
        if self.Plane_Irtifa < 2000 or self.Plane_Irtifa > 7000:
            self.reward -= 15.0
        if self.Plane_Irtifa < 600 or self.Plane_Irtifa > 10000:
            self.reward -= 100.0  # Large penalty for dangerous altitudes

        # 5. Missile firing logic
        if self.now_missile_state:
            if self.missile1_state and not self.Ally_target_locked:
                self.reward -= 8.0  # Penalty for firing with no lock
                self.success = -1
                print('Failed to fire: no lock')
            elif self.missile1_state and self.Ally_target_locked :
                self.reward += 10.0  # Bonus for good fire
                print('Successful fire!')
                self.success = 1
                self.fire_success = True
            else:
                self.reward -= 2.0  # Mild penalty for random shots

        # 6. Enemy defeated (terminal reward)
        if self.oppo_health['health_level'] <= 0.1 and self.fire_success:
            self.reward += 250.0
            print('Enemy destroyed!')

        # 7. Time penalty (small per step to encourage efficiency)
        self.reward -= 0.05

        reward = self.reward
        return reward

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
        # Altitude limits
        if self.Plane_Irtifa < 500 or self.Plane_Irtifa > 10000:
            self.done = True
            # print(f"Episode terminated: Altitude limit (altitude: {self.Plane_Irtifa:.1f}m)")

        # Enemy health
        if self.oppo_health['health_level'] <= 0:
            self.done = True
            self.episode_success = True
            # print(f"Episode terminated: Enemy destroyed! 🎉")

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
            file.write(source_code1)
            file.write(' ')
            file.write(source_code2)
            file.write(' ')
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