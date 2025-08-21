# HarfangEnv_GYM.py
"""
Harfang Gym-style environment with vectorized missile tails.

Major points
------------
- Preserves existing reward/termination/tactics. No numerical changes.
- Returns gym-style step tuple: (obs, reward, done, info).
  * info includes: opponent_obs, success (step flag), episode_success, now_missile_state, missile1_state, n_missile1_state
- Observations are **purely numeric** (stable_baselines-friendly).
  * Missile data is vectorized into fixed-length packs: [present, mx, my, mz, heading_deg] * MAX_TRACKED_MISSILES
  * Absolute missile positions are in meters (unchanged assumption used by evasive logic).
- Adds observation_space with dynamic size computed at reset.
- Centralizes primitive actions in Action_Helper; SimpleEnemy extends base env for a minimalist adversary.

State layout (indices)
----------------------
0..2     : Pos_Diff = [dx, dy, dz] from Ally POV (normalized by 1e4)
3..5     : Ally Euler angles (normalized by NormStates["Plane_Euler_angles"])
6        : target_angle (degrees, as provided by sim)
7        : ally target_locked  (1 if True else -1)
8        : ally missile1_state (1 if slot present else -1)
9..11    : Opponent Euler angles (normalized)
12       : Opponent health level (0..1)
13..15   : Ally position (normalized by 1e4)
16..18   : Opponent position (normalized by 1e4)
19       : Ally heading (degrees)
20       : Ally health level (0..1)
21       : Ally pitch attitude (normalized)
22..     : Missile vector packs (MAX_TRACKED_MISSILES * 5 floats):
           For each i in [0..MAX-1]:
             [present, mx, my, mz, heading_deg]
           present ∈ {0,1}, positions in meters (absolute), heading in degrees.
"""
import numpy as np
import gym
import os
import inspect
import random
import math
import re
import time

# harfang SDK
from . import dogfight_client as df
from .constants import *  # NormStates etc.





MAX_TRACKED_MISSILES = 4
MISSILE_PACK_LEN = 5  # [present, mx, my, mz, heading]


class HarfangEnv:
    """
    Harfang air-combat environment.

    Notes
    -----
    * For algorithmic compatibility with stable_baselines, observations contain only numeric
      values, with missile info vectorized into fixed-length packs at the tail.
    * step() returns (obs, reward, done, info), where `info["opponent_obs"]` provides the
      opponent's observation vector for symmetric/scripted policies outside RL use.
    """

    def __init__(self):
        # --- runtime flags/state ---
        self.done = False
        self.loc_diff = 0.0
        self.success = 0
        self.episode_success = False
        self.fire_success = False
        self.now_missile_state = False

        # IDs
        self.Plane_ID_oppo = "ennemy_2"
        self.Plane_ID_ally = "ally_1"

        # Health/bookkeeping
        self.oppo_health = 0.2
        self.ally_health = 1.0

        # Locks/missile slot snapshot
        self.Ally_target_locked = False
        self.n_Ally_target_locked = False
        self.missile1_state = True
        self.n_missile1_state = True

        self.target_angle = 0.0
        self.Plane_Irtifa = 0.0

        # Missile tracker
        self.missile_handler = MissileHandler()

        # Spaces (action fixed; observation computed after first reset)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # observation_space will be finalized at first reset (depends on lengths)
        self.observation_space = None

    # ------------------------------- Public API -------------------------------- #
    def reset(self):
        """
        Reset simulation and return a pair (ally_obs, oppo_obs) for scripts that
        control both sides (rule-based). This mirrors the previous API.

        For gym/RL usage, prefer `reset_gym()` that returns (obs, info).
        """
        self._reset_episode_common()
        state_ally = self._get_observation()          # ally POV
        state_oppo = self._get_enemy_observation()    # opponent POV

        # finalize observation_space on first call
        if self.observation_space is None:
            obs_len = len(state_ally)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32
            )

        return state_ally, state_oppo

    def reset_gym(self):
        """
        Gym-style reset returning (obs, info).
        """
        ally_obs, oppo_obs = self.reset()
        return ally_obs, {"opponent_obs": oppo_obs}

    def step(self, action_ally, action_enemy):
        """
        Apply ally/opponent actions. Returns gym-style 4-tuple:

        Returns
        -------
        obs : np.ndarray
            Ally observation after step.
        reward : float
        done : bool
        info : dict
            - "opponent_obs": opponent observation after step
            - "success": per-step success flag used by previous code
            - "episode_success": episode-level success
            - "now_missile_state", "missile1_state", "n_missile1_state"
        """
        self._apply_action(action_ally, action_enemy)
        self.missile_handler.refresh_missiles()

        n_state = self._get_observation()
        n_state_oppo = self._get_enemy_observation()

        self._get_reward(self.state, action_ally, n_state)
        self.state = n_state
        self.oppo_state = n_state_oppo
        self._get_termination()

        info = {
            "opponent_obs": n_state_oppo,
            "success": self.success,
            "episode_success": self.episode_success,
            "now_missile_state": self.now_missile_state,
            "missile1_state": self.missile1_state,
            "n_missile1_state": self.n_missile1_state,
        }
        return n_state, float(self.reward), bool(self.done), info

    # Legacy helper (kept intact)
    def step_test(self, action):
        self._apply_action(action, [0.0, 0.0, 0.0, 0.0])
        n_state = self._get_observation()
        self._get_reward(self.state, action, n_state)
        self.state = n_state
        self._get_termination()
        return n_state, self.reward, self.done, {}, self.now_missile_state, self.missile1_state, self.n_missile1_state, self.Ally_target_locked, self.success

    # ------------------------------- Internals --------------------------------- #
    def _reset_episode_common(self):
        self.Ally_target_locked = False
        self.n_Oppo_target_locked = False
        self.n_Ally_target_locked = False
        self.missile1_state = True
        self.n_missile1_state = True
        self.success = 0
        self.done = False
        self.episode_success = False
        self.fire_success = False
        self.now_missile_state = False

        # Makine durumlarını sıfırla
        self._reset_machine()
        self._reset_missile()
        self.missile_handler.refresh_missiles()

        # 🔧 ESKİ DAVRANIŞIN GERİ GETİRİLMESİ — hedef ataması (KİLİT İÇİN ŞART)
        df.set_target_id(self.Plane_ID_ally, self.Plane_ID_oppo)  # ally hedefi: enemy
        df.set_target_id(self.Plane_ID_oppo, self.Plane_ID_ally)  # enemy hedefi: ally

        # State önbellekleri
        self.state = None
        self.oppo_state = None

    def _apply_action(self, action_ally, action_enemy):
        # Ally controls
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        # Opponent controls
        df.set_plane_pitch(self.Plane_ID_oppo, float(action_enemy[0]))
        df.set_plane_roll(self.Plane_ID_oppo, float(action_enemy[1]))
        df.set_plane_yaw(self.Plane_ID_oppo, float(action_enemy[2]))

        # Missile fire handling (finalized to use actual available slots per side)
        self.now_missile_state = False
        if float(action_ally[3]) > 0.0:
            ally_unfired_slots = self._unfired_slots(self.Plane_ID_ally)
            if ally_unfired_slots:
                df.fire_missile(self.Plane_ID_ally, min(ally_unfired_slots))
                self.now_missile_state = True

        if float(action_enemy[3]) > 0.0:
            oppo_unfired_slots = self._unfired_slots(self.Plane_ID_oppo)
            if oppo_unfired_slots:
                df.fire_missile(self.Plane_ID_oppo, min(oppo_unfired_slots))

        df.update_scene()

    @staticmethod
    def _unfired_slots(plane_id):
        """
        Return a list of integer slot indices that currently hold an unfired missile
        for the given plane_id.
        """
        # Slots e.g. ["ally_1AIM_SL0", "ally_1Meteor1", ...]
        slots = df.get_machine_missiles_list(plane_id)
        slot_state = df.get_missiles_device_slots_state(plane_id).get("missiles_slots", [])
        unfired = []
        for i, present in enumerate(slot_state):
            if i < len(slots) and bool(present):
                # Unfired missiles are those with position [0,0,0]
                missile_id_guess = MissileHandler.slotid_to_missileid(slots[i])
                try:
                    st = df.get_missile_state(missile_id_guess)
                    if list(st.get("position", [0.0, 0.0, 0.0])) == [0.0, 0.0, 0.0]:
                        unfired.append(i)
                except Exception:
                    # if state cannot be read, still allow fire by slot index
                    unfired.append(i)
        return unfired

    def _get_reward(self, state, action, n_state):
        # Preserved reward function
        self.reward = 0
        self.success = 0
        self._get_loc_diff()

        self.reward -= (0.0001 * self.loc_diff)
        self.reward -= self.target_angle * 10

        if self.Plane_Irtifa < 2000:
            self.reward -= 4
        if self.Plane_Irtifa > 7000:
            self.reward -= 4

        if self.now_missile_state is True:
            self.reward -= 8
            if self.missile1_state and (self.Ally_target_locked is False):
                self.success = -1
                print('failed to fire')
            elif self.missile1_state and (self.Ally_target_locked is True):
                print('successful to fire')
                self.success = 1
                self.fire_success = True
            else:
                self.reward -= 0

        if self.oppo_health['health_level'] <= 0.1 and self.fire_success:
            self.reward += 600
            print('enemy have fallen')

    def _get_termination(self):
        if self.Plane_Irtifa < 500 or self.Plane_Irtifa > 10000:
            self.done = True
        if self.oppo_health['health_level'] <= 0:
            self.done = True
            self.episode_success = True
        if self.ally_health['health_level'] < 1.0:
            self.done = True

    def _reset_machine(self):
        df.reset_machine("ally_1")
        df.reset_machine("ennemy_2")
        df.set_health("ennemy_2", 0.2)
        df.set_health("ally_1", 1.0)
        self.oppo_health = 0.2
        self.ally_health = 1.0

        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)

        df.set_plane_thrust(self.Plane_ID_ally, 1.0)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def _reset_missile(self):
        self.now_missile_state = False
        df.rearm_machine(self.Plane_ID_ally)
        df.rearm_machine(self.Plane_ID_oppo)

    def _get_loc_diff(self):
        self.loc_diff = (
            ((self.Aircraft_Loc[0] - self.Oppo_Loc[0]) ** 2) +
            ((self.Aircraft_Loc[1] - self.Oppo_Loc[1]) ** 2) +
            ((self.Aircraft_Loc[2] - self.Oppo_Loc[2]) ** 2)
        ) ** 0.5

    # ------------------------------- Observations ------------------------------- #
    def _get_observation(self):
        """
        Ally POV observation vector. See module docstring for indices.
        """
        plane = df.get_plane_state(self.Plane_ID_ally)
        oppo = df.get_plane_state(self.Plane_ID_oppo)

        Plane_Pos = [plane["position"][0] / NormStates["Plane_position"],
                     plane["position"][1] / NormStates["Plane_position"],
                     plane["position"][2] / NormStates["Plane_position"]]
        Plane_Euler = [plane["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
                       plane["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                       plane["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]
        Plane_Heading = plane["heading"]  # degrees
        Plane_Pitch_Att = plane["pitch_attitude"] / NormStates["Plane_pitch_attitude"]
        Plane_Roll_Att = plane["roll_attitude"] / NormStates["Plane_roll_attitude"]  # noqa: F841

        Oppo_Pos = [oppo["position"][0] / NormStates["Plane_position"],
                    oppo["position"][1] / NormStates["Plane_position"],
                    oppo["position"][2] / NormStates["Plane_position"]]
        Oppo_Euler = [oppo["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
                      oppo["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                      oppo["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]

        self.Plane_Irtifa = plane["position"][1]
        self.Aircraft_Loc = plane["position"]
        self.Oppo_Loc = oppo["position"]

        # Locks/flags
        self.Ally_target_locked = self.n_Ally_target_locked
        self.n_Ally_target_locked = plane["target_locked"]
        locked = 1 if self.n_Ally_target_locked else -1

        self.Oppo_target_locked = self.n_Oppo_target_locked
        self.n_Oppo_target_locked = oppo["target_locked"]
        oppo_locked = 1 if self.n_Oppo_target_locked else -1  # noqa: F841 (kept for symmetry)

        target_angle = plane['target_angle']
        self.target_angle = target_angle

        Pos_Diff = [Oppo_Pos[0] - Plane_Pos[0],
                    Oppo_Pos[1] - Plane_Pos[1],
                    Oppo_Pos[2] - Plane_Pos[2]]

        self.oppo_health = df.get_health(self.Plane_ID_oppo)
        self.ally_health = df.get_health(self.Plane_ID_ally)
        ally_hea = self.ally_health['health_level']
        oppo_hea = self.oppo_health['health_level']

        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_ally)
        self.missile1_state = self.n_missile1_state
        self.n_missile1_state = Missile_state["missiles_slots"][0] if Missile_state["missiles_slots"] else False
        missile1_state_val = 1 if self.n_missile1_state else -1

        # Vectorize incoming enemy missiles (absolute meters)
        missile_vec = self._vectorize_missiles(self.get_enemy_missile_vector())

        States = np.concatenate((
            Pos_Diff,                      # 0..2
            Plane_Euler,                   # 3..5
            [target_angle],                # 6
            [locked],                      # 7
            [missile1_state_val],          # 8
            Oppo_Euler,                    # 9..11
            [oppo_hea],                    # 12
            Plane_Pos,                     # 13..15
            Oppo_Pos,                      # 16..18
            [Plane_Heading],               # 19
            [ally_hea],                    # 20
            [Plane_Pitch_Att],             # 21
            missile_vec                    # 22..
        ), axis=None).astype(np.float32)

        self.state = States
        return States

    def _get_enemy_observation(self):
        """
        Opponent POV observation vector (same layout semantics as ally POV).
        """
        plane = df.get_plane_state(self.Plane_ID_oppo)  # enemy self
        oppo = df.get_plane_state(self.Plane_ID_ally)   # ally as opponent from enemy POV

        Plane_Pos = [plane["position"][0] / NormStates["Plane_position"],
                     plane["position"][1] / NormStates["Plane_position"],
                     plane["position"][2] / NormStates["Plane_position"]]
        Oppo_Pos = [oppo["position"][0] / NormStates["Plane_position"],
                    oppo["position"][1] / NormStates["Plane_position"],
                    oppo["position"][2] / NormStates["Plane_position"]]

        Plane_Euler = [plane["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
                       plane["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                       plane["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]
        Oppo_Euler = [oppo["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
                      oppo["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                      oppo["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]

        Plane_Heading = plane["heading"]
        Plane_Pitch_Att = plane["pitch_attitude"] / NormStates["Plane_pitch_attitude"]  # noqa: F841
        Plane_Roll_Att = plane["roll_attitude"] / NormStates["Plane_roll_attitude"]      # noqa: F841

        n_Oppo_target_locked = plane["target_locked"]
        locked = 1 if n_Oppo_target_locked else -1

        n_Ally_target_locked = oppo["target_locked"]  # noqa: F841 (kept for parity)
        target_angle = plane["target_angle"]

        Pos_Diff = [Oppo_Pos[0] - Plane_Pos[0],
                    Oppo_Pos[1] - Plane_Pos[1],
                    Oppo_Pos[2] - Plane_Pos[2]]

        missile_vec = self._vectorize_missiles(self.get_ally_missile_vector())

        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_oppo)
        missile1_state_val = 1 if (Missile_state["missiles_slots"] and Missile_state["missiles_slots"][0]) else -1

        oppo_health = df.get_health(self.Plane_ID_oppo)
        oppo_hea = oppo_health['health_level']

        States = np.concatenate((
            Pos_Diff,            # 0..2
            Plane_Euler,         # 3..5
            [target_angle],      # 6
            [locked],            # 7
            [missile1_state_val],# 8
            Oppo_Euler,          # 9..11 (ally euler)
            [oppo_hea],          # 12 (enemy self health)
            Plane_Pos,           # 13..15 (enemy self pos)
            Oppo_Pos,            # 16..18 (ally pos)
            [Plane_Heading],     # 19 (enemy heading)
            missile_vec          # 20..
        ), axis=None).astype(np.float32)

        self.oppo_state = States
        return States

    # --------------------------- Missile vector helpers -------------------------- #
    def _vectorize_missiles(self, missiles):
        """
        Convert a list of missile dicts into a fixed-length float vector:
        [present, mx, my, mz, heading_deg] * MAX_TRACKED_MISSILES
        Using absolute position values in meters to preserve original evasive logic.
        """
        vec = np.zeros((MAX_TRACKED_MISSILES * MISSILE_PACK_LEN,), dtype=np.float32)
        if not missiles:
            return vec
        count = min(len(missiles), MAX_TRACKED_MISSILES)
        for i in range(count):
            m = missiles[i]
            base = i * MISSILE_PACK_LEN
            pos = m.get("position", [0.0, 0.0, 0.0])
            vec[base + 0] = 1.0
            vec[base + 1] = float(pos[0])
            vec[base + 2] = float(pos[1])
            vec[base + 3] = float(pos[2])
            vec[base + 4] = float(m.get("heading", 0.0))
        return vec

    def _parse_missiles_from_state(self, state):
        """
        Inverse of _vectorize_missiles() for use by action helpers that expect a list
        of dict missiles from the state tail. This preserves previous behavior while
        keeping the observation fully numeric for RL.
        """
        missiles = []
        start = 22
        if len(state) <= start:
            return missiles
        for i in range(MAX_TRACKED_MISSILES):
            base = start + i * MISSILE_PACK_LEN
            if base + 4 >= len(state):
                break
            present = state[base + 0] > 0.5
            if not present:
                continue
            mx = float(state[base + 1]); my = float(state[base + 2]); mz = float(state[base + 3])
            hdg = float(state[base + 4])
            if (mx, my, mz) == (0.0, 0.0, 0.0):
                continue
            missiles.append({
                "missile_id": f"M{i}",
                "position": [mx, my, mz],
                "heading": hdg,
            })
        return missiles

    # Public missile queries (unchanged semantics)
    def get_enemy_missile_vector(self):
        """List of dicts for all current enemy missiles."""
        self.missile_handler.refresh_missiles()
        missile_info_list = []
        for mid in sorted(self.missile_handler.enemy_missiles):
            try:
                state = df.get_missile_state(mid)
                if not state.get("wreck", False):
                    missile_info_list.append({
                        "missile_id": mid,
                        "position": list(state["position"][:3]),
                        "heading": state.get("heading", 0),
                    })
            except Exception:
                pass
        return missile_info_list

    def get_ally_missile_vector(self):
        """List of dicts for all current ally missiles."""
        self.missile_handler.refresh_missiles()
        missile_info_list = []
        for mid in sorted(self.missile_handler.ally_missiles):
            try:
                state = df.get_missile_state(mid)
                if not state.get("wreck", False):
                    missile_info_list.append({
                        "missile_id": mid,
                        "position": list(state["position"][:3]),
                        "heading": state.get("heading", 0),
                    })
            except Exception:
                pass
        return missile_info_list

    # ----------------------------- Utility / Logging ---------------------------- #
    def get_pos(self):
        plane_state = df.get_plane_state(self.Plane_ID_ally)
        return np.array(plane_state["position"][:3])

    def get_oppo_pos(self):
        plane_state = df.get_plane_state(self.Plane_ID_oppo)
        return np.array(plane_state["position"][:3])

    def save_parameters_to_txt(self, log_dir):
        source_code1 = inspect.getsource(self._get_reward)
        source_code2 = inspect.getsource(self._reset_machine)
        source_code3 = inspect.getsource(self._get_termination)
        filename = os.path.join(log_dir, "log2.txt")
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(source_code1 + '\n')
            file.write(source_code2 + '\n')
            file.write(source_code3 + '\n')

    # --------------------------- Expert-data helpers ---------------------------- #
    def get_loc_diff(self, state):
        loc_diff = (((state[0] * 10000) ** 2) + ((state[1] * 10000) ** 2) + ((state[2] * 10000) ** 2)) ** 0.5
        return loc_diff

    def get_reward(self, state, action, n_state):
        # preserved copy of the compact reward for data extraction
        reward = 0
        step_success = 0
        loc_diff = self.get_loc_diff(n_state)
        reward -= (0.0001 * loc_diff)
        reward -= (n_state[6]) * 10
        if action[-1] > 0:
            reward -= 8
            if state[8] > 0 and state[7] < 0:
                step_success = -1
            elif state[8] > 0 and state[7] > 0:
                step_success = 1
            else:
                reward -= 0
        if n_state[-1] < 0.1:
            reward += 600
        return reward, step_success

    def get_termination(self, state):
        return bool(state[-1] <= 0.1)


# ----------------------------- Missile management ------------------------------ #

class MissileHandler:
    MISSILE_TYPES = ["AIM_SL", "Meteor", "Karaoke", "Mica", "S400", "Sidewinder"]

    def __init__(self):
        self.ally_id = "ally_1"
        self.enemy_id = "ennemy_2"
        self.ally_slots = []
        self.enemy_slots = []
        self.ally_missiles = set()
        self.enemy_missiles = set()
        self.refresh_missiles()

    def refresh_missiles(self):
        self.ally_slots = df.get_machine_missiles_list(self.ally_id)
        self.ally_slot_states = df.get_missiles_device_slots_state(self.ally_id).get("missiles_slots", [])
        self.ally_missiles = self._get_current_missiles("ally")

        self.enemy_slots = df.get_machine_missiles_list(self.enemy_id)
        self.enemy_slot_states = df.get_missiles_device_slots_state(self.enemy_id).get("missiles_slots", [])
        self.enemy_missiles = self._get_current_missiles("ennemy")

        self.ally_missiles = set([mid for mid in self.ally_missiles if not self.is_missile_wreck(mid)])
        self.enemy_missiles = set([mid for mid in self.enemy_missiles if not self.is_missile_wreck(mid)])

    def _get_current_missiles(self, side_str):
        missiles = df.get_missiles_list()
        filtered = []
        for mid in missiles:
            if any(t in mid for t in self.MISSILE_TYPES) and mid.startswith(side_str):
                filtered.append(mid)
        return set(filtered)

    @staticmethod
    def slotid_to_missileid(slot_id):
        m = re.match(r"(\w+?)_?(AIM_SL|Meteor|Karaoke|Mica|S400|Sidewinder)(\d+)", slot_id)
        if m:
            prefix, mtype, num = m.groups()
            return f"{prefix}-{mtype}-{num}"
        return slot_id.replace("_", "-")

    @staticmethod
    def missileid_to_slotid(missile_id):
        parts = missile_id.split('-')
        if len(parts) >= 3:
            return f"{parts[0]}{parts[1]}{parts[2]}"
        return missile_id.replace("-", "")

    def missile_slot_status(self, plane_id):
        slots = df.get_machine_missiles_list(plane_id)
        slot_state = df.get_missiles_device_slots_state(plane_id).get("missiles_slots", [])
        results = []
        for idx, slot_id in enumerate(slots):
            present = slot_state[idx] if idx < len(slot_state) else False
            missile_id_guess = self.slotid_to_missileid(slot_id)
            missile_state = None
            wreck = None
            pos = None
            if present:
                try:
                    missile_state = df.get_missile_state(missile_id_guess)
                    wreck = missile_state.get("wreck", None)
                    pos = missile_state.get("position", None)
                except Exception:
                    wreck = None
            results.append({
                "slot_idx": idx,
                "slot_id": slot_id,
                "missile_id": missile_id_guess,
                "slot_active": present,
                "wreck": wreck,
                "position": pos
            })
        return results

    def is_missile_wreck(self, missile_id):
        try:
            missile_state = df.get_missile_state(missile_id)
            return missile_state.get("wreck", False)
        except Exception:
            return True

    def track_all_missiles(self, side="ally", steps=10, print_missing=True):
        self.refresh_missiles()
        missile_ids = list(self.ally_missiles) if side == "ally" else list(self.enemy_missiles)
        if not missile_ids:
            print(f"[{side.upper()}] No active missiles.")
            return
        for mid in missile_ids:
            for i in range(steps):
                try:
                    state = df.get_missile_state(mid)
                    pos = state.get("position", None)
                    wreck = state.get("wreck", None)
                    if pos is not None and wreck is not True:
                        print(f"  [{side.upper()} MISSILE:{mid}] Step {i:02d}: x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.2f}")
                    else:
                        if print_missing:
                            print(f"  [{side.upper()} MISSILE:{mid}] Step {i:02d}: (wreck/position missing)")
                        break
                except Exception:
                    if print_missing:
                        print(f"  [{side.upper()} MISSILE:{mid}] Step {i:02d}: (state unavailable)")
                    break

    def missile_summary(self):
        self.refresh_missiles()
        print(f"ALLY missiles: {sorted(self.ally_missiles)}")
        print(f"ENEMY missiles: {sorted(self.enemy_missiles)}")
        print("Ally slot state:", self.missile_slot_status(self.ally_id))
        print("Enemy slot state:", self.missile_slot_status(self.enemy_id))


# --------------------------- Minimal enemy environment ------------------------- #
class SimpleEnemy(HarfangEnv):
    """
    Minimal adversary wrapper that still respects the finalized slot-based firing.
    """

    def __init__(self):
        super(SimpleEnemy, self).__init__()
        self.has_fired = False

    def _apply_action(self, action_ally, action_enemy):
        # Apply basic controls
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        df.set_plane_pitch(self.Plane_ID_oppo, float(action_enemy[0]))
        df.set_plane_roll(self.Plane_ID_oppo, float(action_enemy[1]))
        df.set_plane_yaw(self.Plane_ID_oppo, float(action_enemy[2]))

        # Slot-based fires (ally)
        if float(action_ally[3]) > 0.0:
            ally_unfired_slots = self._unfired_slots(self.Plane_ID_ally)
            if ally_unfired_slots:
                df.fire_missile(self.Plane_ID_ally, min(ally_unfired_slots))
                self.now_missile_state = True
                print(" === ally fired missile! ===")
        else:
            self.now_missile_state = False

        # Slot-based fires (enemy)
        if float(action_enemy[3]) > 0.0:
            oppo_unfired_slots = self._unfired_slots(self.Plane_ID_oppo)
            if oppo_unfired_slots:
                df.fire_missile(self.Plane_ID_oppo, min(oppo_unfired_slots))
                print(" === enemy fired missile! ===")

        df.update_scene()
