import numpy as np
import math
from .constants import *  # NormStates etc.

MAX_TRACKED_MISSILES = 4
MISSILE_PACK_LEN = 5  # [present, mx, my, mz, heading]


class ActionHelper:
    """
    Centralized low-level control primitives used by Agents.
    Supports both dict (new) and array (legacy) states.
    """

    # -------------------------- Internal missile parser ------------------------- #
    @staticmethod
    def _parse_missiles(state):
        """
        Parse missiles from state (dict or array) into list of dicts.
        """
        missiles = []
        if isinstance(state, dict):
            scalars = []
            i = 0
            while f"missile_{i}" in state:
                scalars.append(float(state[f"missile_{i}"]))
                i += 1
            packs = len(scalars) // MISSILE_PACK_LEN
            for p in range(min(packs, MAX_TRACKED_MISSILES)):
                base = p * MISSILE_PACK_LEN
                present = scalars[base] > 0.5
                if not present:
                    continue
                mx, my, mz = scalars[base + 1], scalars[base + 2], scalars[base + 3]
                hdg = scalars[base + 4]
                if (mx, my, mz) == (0.0, 0.0, 0.0):
                    continue
                missiles.append({
                    "missile_id": f"M{p}",
                    "position": [mx, my, mz],
                    "heading": hdg
                })
        else:  # legacy array
            start = 22
            for i in range(MAX_TRACKED_MISSILES):
                base = start + i * MISSILE_PACK_LEN
                if base + 4 >= len(state):
                    break
                present = state[base] > 0.5
                if not present:
                    continue
                mx, my, mz = map(float, state[base + 1: base + 4])
                if (mx, my, mz) == (0.0, 0.0, 0.0):
                    continue
                missiles.append({"missile_id": f"M{i}", "position": [mx, my, mz]})
        return missiles

    # ----------------------------- Primitive actions --------------------------- #
    def track_cmd(self, state):
        if isinstance(state, dict):
            dx = state.get("pos_diff_x", 0.0) * 10000
            dz = state.get("pos_diff_z", 0.0) * 10000
            dy = state.get("pos_diff_y", 0.0) * 10000
            plane_heading = state.get("plane_heading", 0.0)
            altitude = state.get("altitude", 0.0) * 10000
            target_angle = state.get("target_angle", 0.0)
            plane_pitch_norm = state.get("plane_roll", 0.0)  # Euler 0 index? (env mapping)
        else:  # array
            dx, dy, dz = state[0] * 10000, state[2] * 10000, state[1] * 10000
            plane_heading = state[19]
            altitude = state[14] * 10000
            target_angle = state[6]
            plane_pitch_norm = state[3]

        # Bearing calc
        angle_to_enemy = np.degrees(np.arctan2(dx, dy))
        relative_bearing = (angle_to_enemy - plane_heading + 180) % 360 - 180

        # Pitch
        if altitude < 1200:
            pitch_cmd = -0.3
        elif altitude > 8000:
            pitch_cmd = 0.3
        else:
            plane_pitch = math.degrees(plane_pitch_norm * math.pi) * (-1)
            horiz_dist = np.sqrt(dx ** 2 + dy ** 2)
            pitch_to_target = np.degrees(np.arctan2(dz, horiz_dist))
            relative_pitch = (pitch_to_target - plane_pitch + 90) % 180 - 90
            gain = np.interp(horiz_dist, [0, 800], [1.5, 1.1])
            if horiz_dist < 800:
                pitch_cmd = np.clip(-0.03 * relative_pitch * gain / 30, -1, 1)
            else:
                pitch_cmd = np.clip((dz / 10000.0) * -0.2, -1, 1)
            if abs(relative_pitch) > 0.5:
                pitch_cmd = -0.25 if relative_pitch > 0 else 0.25

        roll_cmd = 0.0
        yaw_gain = 0.03 if abs(relative_bearing) < 10 else 0.06
        yaw_cmd = np.clip(relative_bearing * yaw_gain, -1.0, 1.0)
        return [float(pitch_cmd), float(roll_cmd), float(yaw_cmd), -1.0]

    def evade_cmd(self, state):
        missiles = self._parse_missiles(state)
        altitude = state.get("altitude", 0.0) * 10000 if isinstance(state, dict) else state[14] * 10000
        heading = state.get("plane_heading", 0.0) if isinstance(state, dict) else state[19]

        # no threat → level flight
        if not missiles:
            return [0.0, 0.0, 0.0, 0.0]

        # take first missile
        m = missiles[0]
        mx, my, mz = m["position"]
        # simple: bank away
        yaw_cmd = -0.6 if mx > 0 else 0.6
        roll_cmd = yaw_cmd * 0.5
        pitch_cmd = -0.1 if altitude < 2000 else 0.1
        return [pitch_cmd, roll_cmd, yaw_cmd, -1.0]

    def climb_cmd(self, state):
        altitude = state.get("altitude", 0.0) * 10000 if isinstance(state, dict) else state[14] * 10000
        pitch_att = state.get("plane_pitch_att", 0.0) * 180 if isinstance(state, dict) else state[21] * 180
        pitch_cmd = -0.1
        if altitude > 7000:
            pitch_cmd = 0.25
        if pitch_att > 75:
            pitch_cmd = 0.2
        elif pitch_att < -75:
            pitch_cmd = -0.2
        return [pitch_cmd, 0.0, 0.0, 0.0]

    def fire_cmd(self, state):
        locked = state.get("locked", -1) if isinstance(state, dict) else state[7]
        fire_cmd = 1.0 if locked > 0 else 0.0
        return [0.0, 0.0, 0.0, fire_cmd]

    def enemys_track_cmd(self, state):
        return self.track_cmd(state)
