from dataclasses import dataclass
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from hirl.environments.HarfangEnv_GYM_new import HarfangEnv, SimpleEnemy
from action_helper import ActionHelper


# ----------------------------- Thin action wrapper ----------------------------- #
class Agents:
    """
    Thin facade that delegates primitive actions to ActionHelper to avoid duplication.
    Do NOT change the underlying control numerics here.
    """

    def __init__(self):
        self.action_helper = ActionHelper()

    def update(self, state):
        self.state = state

    @staticmethod
    def _has_incoming_threat(state) -> bool:
        """
        Returns True if any missile 'present' flag is 1 in the vectorized missile tail.
        See HarfangEnv_GYM.MISSILE_PACK_LEN.
        """
        MISSILE_START = 22
        MISSILE_PACK_LEN = 5
        if len(state) <= MISSILE_START:
            return False
        # Scan packs: [present, mx, my, mz, heading]
        for i in range(MISSILE_START, len(state), MISSILE_PACK_LEN):
            present = state[i]
            if present > 0.5:
                # if actual position is non-zero, consider it a threat
                mx, my, mz = state[i + 1: i + 4]
                if (mx, my, mz) != (0.0, 0.0, 0.0):
                    return True
        return False


# ------------------------------ Rule-based Ally -------------------------------- #
class Ally(Agents):
    """
    Simple rule-based ally agent.

    Behaviors (preserved):
    - Early 'track' phase, then switch based on altitude/threat/distance.
    - Fire only under lock, spacing shots by a cooldown (steps_between_fires).
    - Evade when missile threat is present and close (< 8 km).
    """

    def __init__(self, steps_between_fires=600, debug=False):
        super().__init__()
        self.state = None
        self._last_fire_step = -10_000
        self._step = 0
        self.debug = bool(debug)
        self.steps_between_fires = int(steps_between_fires)

    def behave(self):
        assert self.state is not None, "Call update(state) before behave()."
        s = self.state

        # Relative vector to opponent (meters)
        dx, dy, dz = s[0] * 10000.0, s[2] * 10000.0, s[1] * 10000.0
        distance_m = float(np.sqrt(dx * dx + dy * dy + dz * dz))

        altitude_m = s[14] * 10000.0
        locked = bool(s[7] > 0)

        # Missile threat: see env docs — present flag at the start of each missile pack
        threat_detected = self._has_incoming_threat(s)

        # Phased behavior (kept structure/numbers)
        if self._step < 300:
            command = "track"
        else:
            if threat_detected and distance_m < 8000.0:
                command = "evade"
            else:
                if altitude_m < 1000.0:
                    command = "track"  # original path prefers track here
                else:
                    # Fire gate: distance + lock + cooldown
                    if (distance_m < 3000.0) and locked and (self._step - self._last_fire_step) > self.steps_between_fires:
                        command = "fire"
                        self._last_fire_step = self._step
                    else:
                        command = "track"

        if self.debug:
            print(f"[ALLY] step={self._step} dist={distance_m:.0f}m alt={altitude_m:.0f} lock={int(locked)} threat={threat_detected} -> {command}")

        # Map to action
        if command == "track":
            action = self.action_helper.track_cmd(s)
        elif command == "evade":
            action = self.action_helper.evade_cmd(s)
        elif command == "climb":
            action = self.action_helper.climb_cmd(s)
        elif command == "fire":
            action = self.action_helper.fire_cmd(s)
        else: # DEFAULT action
            action = self.action_helper.track_cmd(s)

        self._step += 1
        return action, command




# ------------------------------ Rule-based Opponent ---------------------------- #
@dataclass
class OppoParams:
    fire_cooldown: int = 450           # adım
    press_range_m: float = 4500.0      # kilitliyken ateş eşiği
    hard_evade_threat_m: float = 9000.0
    beam_threat_m: float = 12000.0
    min_floor_m: float = 1000.0
    climb_target_m: float = 5000.0
    crank_duration: tuple = (40, 70)   # adım aralığı
    beam_duration: tuple = (35, 55)
    egress_duration: tuple = (80, 120)
    rng_seed: int = 2025


class Oppo(Agents):
    def __init__(self, debug=False, fire_cooldown=600):
        super().__init__()
        self.state = None
        self._step = 0
        self.debug = bool(debug)
        self.fire_cooldown = int(fire_cooldown)
        self._last_fire_step = -10_000  # ilk atışa izin

    def behave(self):
        assert self.state is not None, "Call update(state) before behave()."
        s = self.state

        # mesafe/irtifa/lock
        dx, dy, dz = s[0] * 10000.0, s[2] * 10000.0, s[1] * 10000.0
        distance_m = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        altitude_m = s[14] * 10000.0
        locked = bool(s[7] > 0)

        # tehdit algısı (ally füze atınca dolacak)
        threat = Agents._has_incoming_threat(s)

        # --- NET ATEŞ KAPISI (öncelikli) ---
        can_fire = (
            #locked
            (distance_m < 4500.0)
             and ((self._step - self._last_fire_step) > self.fire_cooldown)
        )
        if can_fire:
            command = "fire"
        else:
            # orijinal faz mantığını koruyarak karar ver
            if self._step < 200:
                phase = "approach"
            elif self._step < 500:
                phase = "engage"
            else:
                phase = "maintain"

            if phase == "approach":
                command = "climb" if altitude_m < 4000.0 else "track"
            elif phase == "engage":
                if threat and distance_m < 9000.0:
                    command = "evade"
                else:
                    command = "climb" if altitude_m < 3500.0 else "track"
            else:
                if threat:
                    command = "evade"
                else:
                    command = "climb" if altitude_m < 5000.0 else "track"

        # komutu aksiyona çevir
        if command == "track":
            action = self.action_helper.enemys_track_cmd(s)
        elif command == "evade":
            action = self.action_helper.evade_cmd(s)
        elif command == "climb":
            action = self.action_helper.climb_cmd(s)
        elif command == "fire":
            action = self.action_helper.fire_cmd(s)
            self._last_fire_step = self._step  # cooldown başlat
        else: # DEFAULT action
            action = self.action_helper.enemys_track_cmd(s)

        if self.debug:
            print(f"[OPPO] step={self._step} d={distance_m:.0f}m alt={altitude_m:.0f} lock={int(locked)} thr={int(threat)} -> {command} a={action}")

        self._step += 1
        return action, command