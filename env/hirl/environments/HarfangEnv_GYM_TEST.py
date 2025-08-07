import numpy as np
import gym
import random
import hirl.environments.dogfight_client as df
from hirl.environments.constants import *
import time

# ----- Güvenli Wrapper'lar -----
def safe_get_machine_missiles_list(plane_id, max_attempts=10, wait=0.2):
    for _ in range(max_attempts):
        try:
            missiles = df.get_machine_missiles_list(plane_id)
            if missiles is not None:
                return missiles
        except Exception:
            pass
        time.sleep(wait)
    print(f"[WARN] get_machine_missiles_list({plane_id}): yanıt alınamadı.")
    return []

def safe_get_missile_state(missile_id, max_attempts=10, wait=0.2):
    for _ in range(max_attempts):
        try:
            state = df.get_missile_state(missile_id)
            if state is not None and "position" in state:
                return state
        except Exception:
            pass
        time.sleep(wait)
    print(f"[WARN] get_missile_state({missile_id}): yanıt alınamadı.")
    return None

def safe_get_plane_state(plane_id, max_attempts=10, wait=0.2):
    for _ in range(max_attempts):
        try:
            state = df.get_plane_state(plane_id)
            if state is not None and "position" in state:
                return state
        except Exception:
            pass
        time.sleep(wait)
    print(f"[WARN] get_plane_state({plane_id}): yanıt alınamadı.")
    return None

def safe_get_health(plane_id, max_attempts=5, wait=0.2):
    for _ in range(max_attempts):
        try:
            state = df.get_health(plane_id)
            if state is not None and "health_level" in state:
                return state
        except Exception:
            pass
        time.sleep(wait)
    print(f"[WARN] get_health({plane_id}): yanıt alınamadı.")
    return {"health_level": 0.0}

def safe_get_missiles_device_slots_state(plane_id, max_attempts=5, wait=0.2):
    for _ in range(max_attempts):
        try:
            state = df.get_missiles_device_slots_state(plane_id)
            if state is not None and "missiles_slots" in state:
                return state
        except Exception:
            pass
        time.sleep(wait)
    print(f"[WARN] get_missiles_device_slots_state({plane_id}): yanıt alınamadı.")
    return {"missiles_slots": [False]}

# ----- HarfangEnv -----
class HarfangEnv():
    def __init__(self):
        self.done = False
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float64)
        self.Plane_ID_oppo = "ennemy_2"
        self.Plane_ID_ally = "ally_1"
        self.state = None
        self.reward = 0
        self.success = 0
        self.episode_success = False
        self.fire_success = False
        self.Plane_Irtifa = 0
        self.missile = []
        self.missile1_id = None
        self.loc_diff = 0
        self.target_angle = 0
        self.oppo_health = {"health_level": 0.2}
        self.now_missile_state = False
        self.Ally_target_locked = False
        self.n_Ally_target_locked = False

    def reset(self):
        # Uçak ve düşmanı resetle
        self._reset_machine()
        self._reset_missile()
        self.missile = safe_get_machine_missiles_list(self.Plane_ID_ally)
        self.missile1_id = self.missile[0] if self.missile else None
        self.episode_success = False
        self.fire_success = False
        self.success = 0
        self.done = False
        self.now_missile_state = False
        self.Ally_target_locked = False
        self.n_Ally_target_locked = False
        self.oppo_health = safe_get_health(self.Plane_ID_oppo)
        self.state = self._get_observation()
        return self.state

    def step(self, action):
        self._apply_action(action)
        n_state = self._get_observation()
        self._get_reward(self.state, action, n_state)
        self.state = n_state
        self._get_termination()
        return n_state, self.reward, self.done, {}, self.success

    def _reset_machine(self):
        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine(self.Plane_ID_oppo)
        df.set_health(self.Plane_ID_oppo, 0.2)
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
        time.sleep(0.5)  # slotların hazır olmasını bekle

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))
        df.set_plane_pitch(self.Plane_ID_oppo, 0)
        df.set_plane_roll(self.Plane_ID_oppo, 0)
        df.set_plane_yaw(self.Plane_ID_oppo, 0)

        if float(action_ally[3]) > 0:
            df.fire_missile(self.Plane_ID_ally, 0)
            self.now_missile_state = True
        else:
            self.now_missile_state = False
        df.update_scene()
        time.sleep(0.1)  # server'ın güncellemesini bekle

    def _get_observation(self):
        Plane_state = safe_get_plane_state(self.Plane_ID_ally)
        Oppo_state = safe_get_plane_state(self.Plane_ID_oppo)
        if Plane_state is None or Oppo_state is None:
            print("[WARN] Plane/Oppo state alınamadı!")
            return np.zeros(32)

        # Pozisyon, Euler, heading vs.
        Plane_Pos = [Plane_state["position"][0] / NormStates["Plane_position"],
                     Plane_state["position"][1] / NormStates["Plane_position"],
                     Plane_state["position"][2] / NormStates["Plane_position"]]
        Plane_Euler = [Plane_state["Euler_angles"][i] / NormStates["Plane_Euler_angles"] for i in range(3)]
        Plane_Heading = Plane_state["heading"]

        Oppo_Pos = [Oppo_state["position"][i] / NormStates["Plane_position"] for i in range(3)]
        Oppo_Euler = [Oppo_state["Euler_angles"][i] / NormStates["Plane_Euler_angles"] for i in range(3)]
        Oppo_Heading = Oppo_state["heading"] / NormStates["Plane_heading"]

        # State güncellemeleri
        self.Plane_Irtifa = Plane_state["position"][1]
        self.Ally_target_locked = Plane_state["target_locked"]
        self.oppo_health = safe_get_health(self.Plane_ID_oppo)
        self.target_angle = Plane_state.get("target_angle", 0)

        # Pozisyon farkı
        Pos_Diff = [Oppo_Pos[i] - Plane_Pos[i] for i in range(3)]
        Missile_state = safe_get_missiles_device_slots_state(self.Plane_ID_ally)
        missile1_state = 1 if Missile_state["missiles_slots"][0] else -1

        obs = np.concatenate((
            Pos_Diff, Plane_Euler, [self.target_angle],
            [1 if self.Ally_target_locked else -1], [missile1_state],
            Oppo_Euler, [self.oppo_health.get("health_level", 0.0)],
            Plane_Pos, Oppo_Pos, [Plane_Heading]
        ), axis=None)
        return obs

    def _get_reward(self, state, action, n_state):
        self.reward = 0
        self.success = 0
        # Hedefle aradaki mesafe (loc_diff)
        loc_diff = np.linalg.norm(state[:3]) * 10000
        self.reward -= (0.0001 * loc_diff)
        self.reward -= self.target_angle * 10
        if self.Plane_Irtifa < 2000:
            self.reward -= 4
        if self.Plane_Irtifa > 7000:
            self.reward -= 4
        if self.now_missile_state:
            self.reward -= 8
        if self.oppo_health.get("health_level", 1.0) <= 0.1:
            self.reward += 600
            self.success = 1

    def _get_termination(self):
        if self.Plane_Irtifa < 500 or self.Plane_Irtifa > 10000:
            self.done = True
        if self.oppo_health.get("health_level", 1.0) <= 0:
            self.done = True
            self.episode_success = True

    # RL uyumlu obs
    def get_pos(self):
        Plane_state = safe_get_plane_state(self.Plane_ID_ally)
        return np.array(Plane_state["position"]) if Plane_state else np.zeros(3)

    def get_oppo_pos(self):
        Oppo_state = safe_get_plane_state(self.Plane_ID_oppo)
        return np.array(Oppo_state["position"]) if Oppo_state else np.zeros(3)

