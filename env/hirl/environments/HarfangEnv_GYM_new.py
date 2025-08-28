# HarfangEnv_GYM.py
"""
Harfang Gym-style environment — NOW returns dict observations.

Key updates
-----------
- Observations are returned as dictionaries (per user request).
- observation_space is a gym.spaces.Dict inferred from the first observation.
- Internal self.state / self.oppo_state store dicts.
- Helpers (get_loc_diff/get_reward/get_termination) accept both dict and array
  without breaking future SB3 integration.
- Altitude/Z consistency: position's 2nd component is altitude (index=1) and
  maps to both "plane_z" and "altitude" keys; historically this was state[14].

Missile data layout in the flat tail (legacy, still used internally for vectorization):
[present, mx, my, mz, heading_deg] * MAX_TRACKED_MISSILES
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
    Harfang air-combat environment with dict observations.
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
        self.oppo_health = 0.7
        self.ally_health = 0.7

        # Locks/missile slot snapshot
        self.Ally_target_locked = False
        self.n_Ally_target_locked = False
        self.missile1_state = True
        self.n_missile1_state = True

        self.target_angle = 0.0
        self.Plane_Irtifa = 0.0  # altitude in meters (position[1])

        # Missile tracker
        self.missile_handler = MissileHandler()

        # Spaces
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = None  # will become gym.spaces.Dict after first reset

        # Last states (dicts after first reset)
        self.state = None
        self.oppo_state = None

        self.state_dict = {}

    #     historical attributes
        self.previous_slots = []
        self.fired_missile_names = []

    # ------------------------------- Public API -------------------------------- #
    def reset(self):
        """
        Reset simulation and return a pair (ally_obs_dict, oppo_obs_dict) for scripts
        that control both sides (rule-based).
        """
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

        # Machines & missiles
        self._reset_machine()

        #self._reset_missile() ############

        self.missile_handler.refresh_missiles()

        # Target assignment (lock prerequisite)
        df.set_target_id(self.Plane_ID_ally, self.Plane_ID_oppo)
        df.set_target_id(self.Plane_ID_oppo, self.Plane_ID_ally)

        self.state = None
        self.oppo_state = None

        state_ally = self._get_observation()          # dict (ally POV)
        state_oppo = self._get_enemy_observation()    # dict (opponent POV)

        # finalize observation_space on first call — Dict space derived from keys
        if self.observation_space is None:
            self.observation_space = self._build_obs_space_from(state_ally)

        # cache dicts
        self.state = state_ally
        self.oppo_state = state_oppo

        return state_ally, state_oppo

    def reset_gym(self):
        """
        Gym-style reset returning (obs_dict, info).
        """
        ally_obs, oppo_obs = self.reset()
        return ally_obs, {"opponent_obs": oppo_obs}

    def step(self, action_ally, action_enemy):
        """
        Apply ally/opponent actions.

        Returns
        -------
        obs : dict
            Ally observation after step (dict).
        reward : float
        done : bool
        info : dict
            - "opponent_obs": opponent observation after step (dict)
            - "success": per-step success flag used by previous code
            - "episode_success": episode-level success
            - "now_missile_state", "missile1_state", "n_missile1_state"
        """
        self._apply_action(action_ally, action_enemy)
        self.missile_handler.refresh_missiles()

        n_state = self._get_observation()          # dict
        n_state_oppo = self._get_enemy_observation()  # dict

        # Reward/termination use environment fields; accept dicts for compatibility
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

    # Legacy helper (kept intact semantics; returns dict now)
    def step_test(self, action):
        self._apply_action(action, [0.0, 0.0, 0.0, 0.0])
        n_state = self._get_observation()
        self._get_reward(self.state, action, n_state)
        self.state = n_state
        self._get_termination()
        return (
            n_state,
            self.reward,
            self.done,
            {},
            self.now_missile_state,
            self.missile1_state,
            self.n_missile1_state,
            self.Ally_target_locked,
            self.success,
        )

    # ------------------------------- Internals --------------------------------- #
    def _apply_action(self, action_ally, action_enemy):
        # Ally controls
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        # Opponent controls
        df.set_plane_pitch(self.Plane_ID_oppo, float(action_enemy[0]))
        df.set_plane_roll(self.Plane_ID_oppo, float(action_enemy[1]))
        df.set_plane_yaw(self.Plane_ID_oppo, float(action_enemy[2]))

        # Missile fire handling (slot-based)
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
        slots = df.get_machine_missiles_list(plane_id)
        slot_state = df.get_missiles_device_slots_state(plane_id).get("missiles_slots", [])
        unfired = []
        for i, present in enumerate(slot_state):
            if i < len(slots) and bool(present):
                missile_id_guess = MissileHandler.slotid_to_missileid(slots[i])
                try:
                    st = df.get_missile_state(missile_id_guess)
                    if list(st.get("position", [0.0, 0.0, 0.0])) == [0.0, 0.0, 0.0]:
                        unfired.append(i)
                except Exception:
                    unfired.append(i)
        return unfired

    def _get_reward(self, state, action, n_state):
        """
        Environment's reward.
        """
        self.reward = 0
        self.success = 0

        # --- Damage-based rewards (proportional to health delta) --------------------------------------
        prev_ally = float(state.get("ally_health", 1.0))
        curr_ally = float(n_state.get("ally_health", 1.0))
        prev_oppo = float(state.get("oppo_health", 1.0))
        curr_oppo = float(n_state.get("oppo_health", 1.0))

        # If your health is in [0,1], keep SCALE=1.0. If it's [0,100], set SCALE=100.0
        SCALE = 1.0
        EPS = 1e-6  # ignore tiny numerical noise
        W_ENEMY_HIT = 400.0  # +reward if *enemy* loses all health in one step (scaled)
        W_ALLY_HIT = 500.0  # -penalty if *ally* loses all health in one step (scaled)
        ALPHA = 1.0  # 1.0 = linear; >1.0 emphasizes big hits; <1.0 smooths spikes

        ally_dmg = max(0.0, (prev_ally - curr_ally))
        oppo_dmg = max(0.0, (prev_oppo - curr_oppo))

        if oppo_dmg > EPS:
            # self.reward += W_ENEMY_HIT * ((oppo_dmg / SCALE) ** ALPHA)
            self.reward += 400 # sabit reward
            print(f"enemy hit: Δ={oppo_dmg:.3f}")

        if ally_dmg > EPS:
            # self.reward -= W_ALLY_HIT * ((ally_dmg / SCALE) ** ALPHA)
            self.reward += -500 # sabit reward
            print(f"ally hit:  Δ={ally_dmg:.3f}")

        # --- Relative Bearing shaping reward ----------------------------------------------------------------
        # K_B = reward scale
        #   ↑ K_B → stronger push to keep enemy nose-on
        #   ↓ K_B → weaker effect, bearing less important
        K_B = 0.12  # keep small vs. hit/kill rewards (≈0.05–0.20)

        # P = front-bias exponent
        #   =1.0 → gentle, forgiving
        #   >1.0 → stricter nose-on discipline
        #   >>3  → too narrow, may tunnel vision
        P = 1.5  # usually 1.0–2.0 is balanced

        # rel_bear: relative bearing of enemy in degrees
        rel_bear = state.get("relative_bearing", 0.0)
        beta = float(rel_bear)

        # reward term: cosine curve peaks at 0° and decays smoothly to 0 at ±180°
        #   (1+cos)/2  ∈ [0,1]  → maps 0°→1, ±180°→0
        # raising to P sharpens the focus on forward sector
        # self.reward += K_B * ((1.0 + math.cos(math.radians(beta))) * 0.5) ** P

        # --- Range shaping reward --------------------------------------------------------------------
        # k_r = reward scale
        #   ↑ k_r → distance to enemy matters more
        #   ↓ k_r → range less important
        k_r = 0.20

        # R* = ideal firing distance (meters, high kill probability band)
        #   Smaller → favors close-in fights
        #   Larger  → favors long-range shots
        R_star = 1500.0

        # σ = tolerance (meters, width of the "good band")f
        #   Smaller → narrow sweet spot, stricter
        #   Larger  → wide band, more forgiving
        sigma = 200.0

        d = float(state.get("distance_to_enemy", 0.0))  # current distance (m)

        # Gaussian reward: peaks at R*, decays smoothly away
        # self.reward += k_r * math.exp(-((d - R_star) ** 2) / (2 * sigma ** 2))

        # ----------------------------------------------------------------------------------------------

        # --- Altitude shaping (Gaussian around preferred A*) -----------------------------------------
        k_a = 0.10  # scale: keep small vs. hit/kill
        A_star = 5000.0  # preferred altitude (m)
        sigma_a = 2000.0  # tolerance (m): smaller=narrow band, larger=forgiving

        alt = float(self.Plane_Irtifa)

        # Bonus peaks at A* (≈k_a) and decays smoothly with distance from A*
        # self.reward += k_a * math.exp(-((alt - A_star) ** 2) / (2.0 * sigma_a ** 2))


        #------Altitude Limits------------
        if alt <  200 or alt > 10000:
            # self.reward -= 1000
            pass
        #---------------------------------

        # ----------------------------------------------------------------------------------------------
        # --- Per-step time cost -----------------
        c_t = 0.01  # small penalty per step
        # self.reward -= c_t
        #---------------------------------------------------

        # --- Too-far penalty -------------------------------------
        D_far = 10000.0  # meters, threshold considered "too far"
        penalty = 2.0  # per-step penalty if beyond D_far

        d = float(state.get("distance_to_enemy", 0.0))

        if d > D_far:
            # self.reward -= penalty
            pass
        #-------------------------------------------------------------------

        if self.oppo_health['health_level'] <= 0.0:
            # self.reward += 1000
            print('enemy have fallen')
            self.success = True

    def _get_termination(self):

        if self.Plane_Irtifa < 500 or self.Plane_Irtifa > 10000:
            self.done = True
        if self.oppo_health['health_level'] <= 0:
            self.done = True
            self.episode_success = True
        if self.ally_health['health_level'] <= 0.2:
            self.done = True

        _, missiles_on_air_count = self.detect_missiles_on_air()

        ally_unfired_slots = self._unfired_slots(self.Plane_ID_ally)
        if len(ally_unfired_slots) == 0:
            if missiles_on_air_count == 0:
                self.done = True

    def detect_missiles_on_air(self):

        current_slots = df.get_machine_missiles_list(self.Plane_ID_ally)
        fired = set(self.previous_slots) - set(current_slots) - {''}
        if len(fired) > 0:
            self.fired_missile_names.append(fired.pop())

        #
        #  run your code here
        if len(self.fired_missile_names) > 0:
            for fired_missile_name in self.fired_missile_names:
                fired_missile_id_guess = MissileHandler.slotid_to_missileid(fired_missile_name)
                fired_missile_st = df.get_missile_state(fired_missile_id_guess)
                fired_missile_wreck = fired_missile_st.get("wreck")

                if fired_missile_wreck:
                    self.fired_missile_names.remove(fired_missile_name)
        #

        self.previous_slots = current_slots
        return self.fired_missile_names, len(self.fired_missile_names)


    def _reset_machine(self):
        df.reset_machine("ally_1")
        df.reset_machine("ennemy_2")
        df.set_health(self.Plane_ID_oppo, 0.7)
        df.set_health(self.Plane_ID_ally, 0.7)
        self.oppo_health = 0.7
        self.ally_health = 0.7

        df.reset_machine_matrix(self.Plane_ID_ally, 0, 4200, 0, 0, 0, 0)
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 3500, -4000, 0, 0, 0)

        df.set_plane_thrust(self.Plane_ID_ally, 1.0)
        df.set_plane_thrust(self.Plane_ID_oppo, 1.0)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 300)
        df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def _reset_missile(self):
        self.now_missile_state = False
        df.rearm_machine(self.Plane_ID_ally)
        df.rearm_machine(self.Plane_ID_oppo)

    def _get_loc_diff(self):
        # Uses absolute positions cached in _get_observation/_get_enemy_observation
        self.loc_diff = (
            ((self.Aircraft_Loc[0] - self.Oppo_Loc[0]) ** 2) +
            ((self.Aircraft_Loc[1] - self.Oppo_Loc[1]) ** 2) +
            ((self.Aircraft_Loc[2] - self.Oppo_Loc[2]) ** 2)
        ) ** 0.5

    # ------------------------------- Observations ------------------------------- #
    def _get_observation(self):
        """
        Ally POV observation (dict).
        """
        plane = df.get_plane_state(self.Plane_ID_ally)
        oppo = df.get_plane_state(self.Plane_ID_oppo)

        Plane_Pos = [
            plane["position"][0] / NormStates["Plane_position"],
            plane["position"][1] / NormStates["Plane_position"],  # altitude (z role), index 1
            plane["position"][2] / NormStates["Plane_position"],
        ]
        Plane_Euler = [
            plane["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
            plane["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
            plane["Euler_angles"][2] / NormStates["Plane_Euler_angles"],
        ]
        Plane_Heading = plane["heading"]  # degrees
        Plane_Pitch_Att = plane["pitch_attitude"] / NormStates["Plane_pitch_attitude"]
        Plane_Roll_Att = plane["roll_attitude"] / NormStates["Plane_roll_attitude"]  # noqa: F841

        Oppo_Pos = [
            oppo["position"][0] / NormStates["Plane_position"],
            oppo["position"][1] / NormStates["Plane_position"],
            oppo["position"][2] / NormStates["Plane_position"],
        ]
        Oppo_Euler = [
            oppo["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
            oppo["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
            oppo["Euler_angles"][2] / NormStates["Plane_Euler_angles"],
        ]

        # Cache absolute positions (meters)
        self.Plane_Irtifa = plane["position"][1]
        self.Aircraft_Loc = plane["position"]
        self.Oppo_Loc = oppo["position"]

        # Locks/flags
        self.Ally_target_locked = self.n_Ally_target_locked
        self.n_Ally_target_locked = plane["target_locked"]
        locked = 1 if self.n_Ally_target_locked else -1

        self.Oppo_target_locked = self.n_Oppo_target_locked
        self.n_Oppo_target_locked = oppo["target_locked"]
        # oppo_locked = 1 if self.n_Oppo_target_locked else -1  # kept for parity if needed

        target_angle = plane['target_angle']
        self.target_angle = target_angle

        # Pos diff in normalized coordinates
        Pos_Diff = [
            Oppo_Pos[0] - Plane_Pos[0],
            Oppo_Pos[1] - Plane_Pos[1],  # this is "z/altitude" lane by convention
            Oppo_Pos[2] - Plane_Pos[2],
        ]

        self.oppo_health = df.get_health(self.Plane_ID_oppo)
        self.ally_health = df.get_health(self.Plane_ID_ally)
        ally_health_ = self.ally_health['health_level']
        oppo_health_ = self.oppo_health['health_level']

        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_ally)
        self.missile1_state = self.n_missile1_state
        self.n_missile1_state = Missile_state["missiles_slots"][0] if Missile_state["missiles_slots"] else False
        missile1_state_val = 1 if self.n_missile1_state else -1

        # Vectorize incoming enemy missiles (absolute meters)
        missile_vec = self._vectorize_missiles(self.get_enemy_missile_vector())

        missile_count = len(self.get_ally_missile_vector())

        # Build flat vector to reuse existing index map where needed
        States = np.concatenate((
            Pos_Diff,                      # 0..2
            Plane_Euler,                   # 3..5
            [target_angle],                # 6
            [locked],                      # 7
            [missile1_state_val],          # 8
            Oppo_Euler,                    # 9..11
            [oppo_health_],                    # 12
            Plane_Pos,                     # 13..15  (index 14 == altitude lane)
            Oppo_Pos,                      # 16..18
            [Plane_Heading],               # 19
            [ally_health_],                    # 20
            [Plane_Pitch_Att],             # 21
            missile_vec                    # 22..
        ), axis=None).astype(np.float32)

        dx, dy, dz = States[0] * 10000, States[2] * 10000, States[1] * 10000
        plan_heading = States[19]
        angle_to_enemy = np.degrees(np.arctan2(dx, dy))
        relative_bearing = (angle_to_enemy - plan_heading + 180) % 360 - 180
        dist_enemy = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        # --- Dictionary mapping (altitude == plane_z == States[14]) ---
        state_dict = {
            "pos_diff_x": States[0],
            "pos_diff_z": States[1],
            "pos_diff_y": States[2],
            "distance_to_enemy" : dist_enemy,

            "plane_roll": States[3],
            "plane_pitch": States[4],
            "plane_yaw": States[5],

            "target_angle": States[6],
            "locked": States[7],
            "missile1_state": States[8],

            "oppo_roll": States[9],
            "oppo_pitch": States[10],
            "oppo_yaw": States[11],

            "oppo_health": States[12],

            "plane_x": States[13],
            "plane_z": States[14],   # altitude lane (normalized)
            "plane_y": States[15],
            "altitude": States[14],  # alias for clarity

            "oppo_x": States[16],
            "oppo_z": States[17],
            "oppo_y": States[18],

            "plane_heading": States[19],
            "ally_health": States[20],
            "plane_pitch_att": States[21],
            "relative_bearing": relative_bearing,

            "missile_count": missile_count
        }

        # Expand missile scalars as missile_0.. missile_{N-1} for dict space stability
        for i in range(len(States) - 22):
            state_dict[f"missile_{i}"] = States[22 + i]

        # Keep internal flat tail parser available if needed by helpers
        self._last_states_flat = States  # optional: for debugging/legacy

        self.state_dict = state_dict

        return state_dict

    def _get_enemy_observation(self):
        """
        Opponent POV observation (dict).
        """
        plane = df.get_plane_state(self.Plane_ID_oppo)  # enemy self
        oppo = df.get_plane_state(self.Plane_ID_ally)   # ally from enemy POV

        Plane_Pos = [
            plane["position"][0] / NormStates["Plane_position"],
            plane["position"][1] / NormStates["Plane_position"],  # altitude lane
            plane["position"][2] / NormStates["Plane_position"],
        ]
        Oppo_Pos = [
            oppo["position"][0] / NormStates["Plane_position"],
            oppo["position"][1] / NormStates["Plane_position"],
            oppo["position"][2] / NormStates["Plane_position"],
        ]

        Plane_Euler = [
            plane["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
            plane["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
            plane["Euler_angles"][2] / NormStates["Plane_Euler_angles"],
        ]
        Oppo_Euler = [
            oppo["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
            oppo["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
            oppo["Euler_angles"][2] / NormStates["Plane_Euler_angles"],
        ]

        Plane_Heading = plane["heading"]
        Plane_Pitch_Att = plane["pitch_attitude"] / NormStates["Plane_pitch_attitude"]

        n_Oppo_target_locked = plane["target_locked"]
        locked = 1 if n_Oppo_target_locked else -1

        # n_Ally_target_locked = oppo["target_locked"]  # parity if needed
        target_angle = plane["target_angle"]

        Pos_Diff = [
            Oppo_Pos[0] - Plane_Pos[0],
            Oppo_Pos[1] - Plane_Pos[1],
            Oppo_Pos[2] - Plane_Pos[2],
        ]

        missile_vec = self._vectorize_missiles(self.get_ally_missile_vector())

        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_oppo)
        missile1_state_val = 1 if (Missile_state["missiles_slots"] and Missile_state["missiles_slots"][0]) else -1


        oppo_health_ = df.get_health(self.Plane_ID_ally)['health_level']
        ally_health_ =  df.get_health(self.Plane_ID_oppo)['health_level']

        States = np.concatenate((
            Pos_Diff,             # 0..2
            Plane_Euler,          # 3..5
            [target_angle],       # 6
            [locked],             # 7
            [missile1_state_val], # 8
            Oppo_Euler,           # 9..11 (ally euler)
            [oppo_health_],           # 12 (enemy self health)
            Plane_Pos,            # 13..15 (enemy self pos)
            Oppo_Pos,             # 16..18 (ally pos)
            [Plane_Heading],      # 19 (enemy heading)
            [ally_health_],           # 20
            [Plane_Pitch_Att],    # 21
            missile_vec           # 22..
        ), axis=None).astype(np.float32)

        oppo_state_dict = {
            "pos_diff_x": States[0],
            "pos_diff_z": States[1],
            "pos_diff_y": States[2],

            "plane_roll": States[3],
            "plane_pitch": States[4],
            "plane_yaw": States[5],

            "target_angle": States[6],
            "locked": States[7],
            "missile1_state": States[8],

            "oppo_roll": States[9],
            "oppo_pitch": States[10],
            "oppo_yaw": States[11],

            "oppo_health": States[12],

            "plane_x": States[13],
            "plane_z": States[14],  # altitude lane
            "plane_y": States[15],
            "altitude": States[14],  # alias

            "oppo_x": States[16],
            "oppo_z": States[17],
            "oppo_y": States[18],

            "plane_heading": States[19],
            "ally_health": States[20],
            "plane_pitch_att": States[21],
        }
        for i in range(len(States) - 22):
            oppo_state_dict[f"missile_{i}"] = States[22 + i]

        return oppo_state_dict

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
            vec[base + 2] = float(pos[1])  # altitude lane
            vec[base + 3] = float(pos[2])
            vec[base + 4] = float(m.get("heading", 0.0))
        return vec

    def _parse_missiles_from_state(self, state):
        """
        Legacy inverse of _vectorize_missiles() for FLAT arrays.
        """
        missiles = []
        start = 22
        if isinstance(state, dict):
            # Prefer dict-aware version
            return self._parse_missiles_from_state_dict(state)
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

    def _parse_missiles_from_state_dict(self, state_dict):
        """
        Dict counterpart: reconstruct missiles from missile_0.. keys.
        """
        scalars = []
        i = 0
        while True:
            key = f"missile_{i}"
            if key not in state_dict:
                break
            scalars.append(float(state_dict[key]))
            i += 1
        missiles = []
        # Group by MISSILE_PACK_LEN
        packs = len(scalars) // MISSILE_PACK_LEN
        for p in range(min(packs, MAX_TRACKED_MISSILES)):
            base = p * MISSILE_PACK_LEN
            present = scalars[base + 0] > 0.5
            if not present:
                continue
            mx, my, mz = scalars[base + 1], scalars[base + 2], scalars[base + 3]
            hdg = scalars[base + 4]
            if (mx, my, mz) == (0.0, 0.0, 0.0):
                continue
            missiles.append({
                "missile_id": f"M{p}",
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
        """
        Compatible distance helper for dict/array observations (normalized axis).
        """
        if isinstance(state, dict):
            dx = float(state.get("pos_diff_x", 0.0)) * 10000.0
            dz = float(state.get("pos_diff_z", 0.0)) * 10000.0  # altitude lane
            dy = float(state.get("pos_diff_y", 0.0)) * 10000.0
            return (dx*dx + dy*dy + dz*dz) ** 0.5
        else:
            # legacy array
            return (((state[0] * 10000) ** 2) + ((state[1] * 10000) ** 2) + ((state[2] * 10000) ** 2)) ** 0.5

    def get_termination(self, state):
        """
        Compact termination for arrays. For dicts, rely on env._get_termination().
        """
        if isinstance(state, dict):
            return False
        return bool(state[-1] <= 0.1)

    # ------------------------------- Spaces ------------------------------------- #
    def _build_obs_space_from(self, obs_dict: dict) -> gym.spaces.Dict:
        """
        Create a gym.spaces.Dict with scalar Boxes for each key in the observation dict.
        """
        return gym.spaces.Dict({
            k: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32)
            for k in obs_dict.keys()
        })


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
# class SimpleEnemy(HarfangEnv):
#     """
#     Minimal adversary wrapper that still respects the finalized slot-based firing.
#     """
#
#     def __init__(self):
#         super(SimpleEnemy, self).__init__()
#         self.has_fired = False
#
#     def _apply_action(self, action_ally, action_enemy):
#         # Apply basic controls
#         df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
#         df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
#         df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))
#
#         df.set_plane_pitch(self.Plane_ID_oppo, float(action_enemy[0]))
#         df.set_plane_roll(self.Plane_ID_oppo, float(action_enemy[1]))
#         df.set_plane_yaw(self.Plane_ID_oppo, float(action_enemy[2]))
#
#         # Slot-based fires (ally)
#         if float(action_ally[3]) > 0.0:
#             ally_unfired_slots = self._unfired_slots(self.Plane_ID_ally)
#             if ally_unfired_slots:
#                 df.fire_missile(self.Plane_ID_ally, min(ally_unfired_slots))
#                 self.now_missile_state = True
#                 print(" === ally fired missile! ===")
#         else:
#             self.now_missile_state = False
#
#         # Slot-based fires (enemy)
#         if float(action_enemy[3]) > 0.0:
#             oppo_unfired_slots = self._unfired_slots(self.Plane_ID_oppo)
#             if oppo_unfired_slots:
#                 df.fire_missile(self.Plane_ID_oppo, min(oppo_unfired_slots))
#                 print(" === enemy fired missile! ===")
#
#         df.update_scene()
