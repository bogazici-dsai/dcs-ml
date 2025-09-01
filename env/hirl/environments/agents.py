from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
# from hirl.environments.HarfangEnv_GYM_new import HarfangEnv  # SimpleEnemy kaldırıldı (modülde yok)
from .action_helper import ActionHelper
import random

# Env ile tutarlı paket uzunlukları (env tarafıyla aynı olmalı)
MISSILE_START = 22          # dict'te bu "scalar tail"in başlangıcı; env mapping'inde 22. indexten itibaren missile_* skalerleri geliyor
MISSILE_PACK_LEN = 5        # [present, mx, my, mz, heading]
MAX_TRACKED_MISSILES = 4


# ----------------------------- Thin action wrapper ----------------------------- #
class Agents:
    """
    Thin facade that delegates primitive actions to ActionHelper to avoid duplication.
    Do NOT change the underlying control numerics here.
    """

    def __init__(self):
        self.action_helper = ActionHelper()
        self.state = None

    def update(self, state):
        self.state = state

    # ---- Missile / threat helpers ------------------------------------------------
    @staticmethod
    def _parse_missiles_from_dict(state_dict):
        """
        Env'in dict mapping'inde füze bilgisi 'missile_0..N' skaler dizisidir.
        Bunları paketlere (present, mx, my, mz, heading) ayırır.
        """
        # missile_0, missile_1, ... şeklinde kaç scalar var topla
        scalars = []
        i = 0
        while True:
            key = f"missile_{i}"
            if key not in state_dict:
                break
            scalars.append(float(state_dict[key]))
            i += 1

        missiles = []
        if not scalars:
            return missiles

        packs = min(len(scalars) // MISSILE_PACK_LEN, MAX_TRACKED_MISSILES)
        for p in range(packs):
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
                "heading": hdg
            })
        return missiles

    @staticmethod
    def _has_incoming_threat(state) -> bool:
        """
        Dict (yeni) ve legacy array state'leri destekler.
        True dönerse aktif bir füze tehdidi vardır.
        """
        # ---- Dict yoluyla ----
        if isinstance(state, dict):
            missiles = Agents._parse_missiles_from_dict(state)
            return len(missiles) > 0

        # ---- Legacy array yoluyla (env'in vektör kuyruğu düzeni) ----
        if isinstance(state, (list, tuple, np.ndarray)) and len(state) > MISSILE_START:
            for i in range(MISSILE_START, len(state), MISSILE_PACK_LEN):
                present = state[i]
                if present > 0.5:
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
        self._last_fire_step = -10_000
        self._step = 0
        self.debug = bool(debug)
        self.steps_between_fires = int(steps_between_fires)

    def behave(self):
        assert self.state is not None, "Call update(state) before behave()."
        s = self.state

        # ---- Mesafe hesabı (dict uyumlu) ----
        if isinstance(s, dict):
            dx = float(s.get("pos_diff_x", 0.0)) * 10000.0
            dz = float(s.get("pos_diff_z", 0.0)) * 10000.0  # altitude lane
            dy = float(s.get("pos_diff_y", 0.0)) * 10000.0
            distance_m = float(np.sqrt(dx * dx + dy * dy + dz * dz))

            altitude_m = float(s.get("altitude", 0.0)) * 10000.0
            locked = bool(float(s.get("locked", -1.0)) > 0.0)
        else:
            # Legacy array fallback (eski eğitim/deney kodları için)
            dx, dy, dz = s[0] * 10000.0, s[2] * 10000.0, s[1] * 10000.0
            distance_m = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            altitude_m = s[14] * 10000.0
            locked = bool(s[7] > 0)

        # Füze tehdidi
        threat_detected = self._has_incoming_threat(s)

        # ---- Karar mantığı (orijinal akış korunarak) ----
        if self._step < 300:
            command = "track"
        else:
            if threat_detected and distance_m < 8000.0:
                command = "evade"
            else:
                if altitude_m < 1000.0:
                    command = "track"
                else:
                    can_fire = (
                        (distance_m < 3000.0)
                        and locked
                        and (self._step - self._last_fire_step) > self.steps_between_fires
                    )
                    if can_fire:
                        command = "fire"
                        self._last_fire_step = self._step
                    else:
                        command = "track"


        if self.debug:
            # print(f"[ALLY] step={self._step} dist={distance_m:.0f}m alt={altitude_m:.0f} lock={int(locked)} threat={threat_detected} -> {command}")
            pass

        # İSTERSEN: Aşağıdaki sabit tırman komutunu kapatabilirsin
        # command = "climb"

        # Komutu aksiyona çevir
        if command == "track":
            action = self.action_helper.track_cmd(s)
        elif command == "evade":
            action = self.action_helper.evade_cmd(s)
        elif command == "climb":
            action = self.action_helper.climb_cmd(s)
            print("ALLY applying CLIMB")
        elif command == "fire":
            action = self.action_helper.fire_cmd(s)
            self._last_fire_step = self._step
        else:
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
    crank_duration: tuple = (40, 70)   # adım aralığı (takipte kalma)
    beam_duration: tuple = (35, 55)    # tehdit varken kısa kaçınma penceresi
    egress_duration: tuple = (80, 120) # yakın tehditte daha uzun kaçınma
    rng_seed: int = 2025


class Oppo(Agents):
    def __init__(self, debug=False, fire_cooldown=600, params: OppoParams | None = None):
        super().__init__()
        self._step = 0
        self.debug = bool(debug)
        self.params = params or OppoParams()
        # Dışarıdan verilen cooldown varsa onu kullan; yoksa parametreyi kullan
        self.fire_cooldown = int(fire_cooldown) if fire_cooldown is not None else int(self.params.fire_cooldown)
        self._last_fire_step = -10_000  # ilk atışa izin

        # Karar pencerelerini hafifçe tutarlı yapmak için RNG ve “komut kilitleyici”
        self._rng = random.Random(self.params.rng_seed)
        self._lock_command_until = -1
        self._locked_command = None

    def _lock_for(self, base_command: str, duration_range: tuple[int, int]):
        """Komutu kısa bir süre sabitleyerek salınımı azalt."""
        dur = self._rng.randint(int(duration_range[0]), int(duration_range[1]))
        self._locked_command = base_command
        self._lock_command_until = self._step + dur

    def behave(self):
        assert self.state is not None, "Call update(state) before behave()."
        s = self.state

        # ---- Mesafe/irtifa/lock (dict uyumlu) ----
        if isinstance(s, dict):
            dx = float(s.get("pos_diff_x", 0.0)) * 10000.0
            dz = float(s.get("pos_diff_z", 0.0)) * 10000.0
            dy = float(s.get("pos_diff_y", 0.0)) * 10000.0
            distance_m = float(np.sqrt(dx * dx + dy * dy + dz * dz))

            altitude_m = float(s.get("altitude", 0.0)) * 10000.0
            locked = bool(float(s.get("locked", -1.0)) > 0.0)
        else:
            dx, dy, dz = s[0] * 10000.0, s[2] * 10000.0, s[1] * 10000.0
            distance_m = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            altitude_m = s[14] * 10000.0
            locked = bool(s[7] > 0)

        # Tehdit algısı
        threat = Agents._has_incoming_threat(s)

        # --- NET ATEŞ KAPISI (parametrelerle) ---
        can_fire = (
            locked
            and (distance_m < float(self.params.press_range_m))
            and ((self._step - self._last_fire_step) > int(self.fire_cooldown))
        )

        # Önceden kilitlenmiş bir komut varsa ve süresi dolmadıysa onu uygula
        if self._locked_command is not None and self._step < self._lock_command_until and not can_fire:
            command = self._locked_command
        else:
            # Faz mantığı (yapıyı bozma): approach / engage / maintain
            if self._step < 200:
                phase = "approach"
            elif self._step < 500:
                phase = "engage"
            else:
                phase = "maintain"

            # İrtifa eşikleri parametrelerden türetildi
            # (mevcut sabitlerin birebir muadili olacak şekilde)
            approach_floor = max(self.params.min_floor_m, self.params.climb_target_m - 1000.0)  # ~4000
            engage_floor = max(self.params.min_floor_m, self.params.climb_target_m - 1500.0)   # ~3500
            maintain_floor = max(self.params.min_floor_m, self.params.climb_target_m)          # ~5000

            # Tehdit eşikleri parametrelerden
            hard_evade_R = float(self.params.hard_evade_threat_m)
            beam_evade_R = float(self.params.beam_threat_m)

            if can_fire:
                command = "fire"
            else:
                if phase == "approach":
                    command = "climb" if altitude_m < approach_floor else "track"

                elif phase == "engage":
                    if threat and distance_m < hard_evade_R:
                        command = "evade"
                        # yakın tehdit: daha uzun kaçınma penceresi
                        self._lock_for("evade", self.params.egress_duration)
                    elif threat and distance_m < beam_evade_R:
                        command = "evade"
                        # orta menzil tehdit: orta kısalıkta kaçınma
                        self._lock_for("evade", self.params.beam_duration)
                    else:
                        command = "climb" if altitude_m < engage_floor else "track"
                        if command == "track":
                            # trak’te kalma süresini hafifçe stabil tut
                            self._lock_for("track", self.params.crank_duration)

                else:  # maintain
                    if threat and distance_m < hard_evade_R:
                        command = "evade"
                        self._lock_for("evade", self.params.egress_duration)
                    elif threat and distance_m < beam_evade_R:
                        command = "evade"
                        self._lock_for("evade", self.params.beam_duration)
                    else:
                        command = "climb" if altitude_m < maintain_floor else "track"
                        if command == "track":
                            self._lock_for("track", self.params.crank_duration)

            # Eğer yeni komut üretildiyse ve kilit süresi geçmişse, _locked_command güncel kalsın;
            # fire komutu için kilitleme uygulamıyoruz (tek atım + cooldown)
            if command == "fire":
                self._locked_command = None
                self._lock_command_until = -1

        # Komutu aksiyona çevir
        if command == "track":
            action = self.action_helper.track_cmd(s)
        elif command == "evade":
            action = self.action_helper.evade_cmd(s)
        elif command == "climb":
            action = self.action_helper.climb_cmd(s)
        elif command == "fire":
            action = self.action_helper.fire_cmd(s)
            self._last_fire_step = self._step  # cooldown
        else:  # DEFAULT
            action = self.action_helper.track_cmd(s)
            command = "track"

        if self.debug:
            # print(f"[OPPO] step={self._step} d={distance_m:.0f}m alt={altitude_m:.0f} lock={int(locked)} thr={int(threat)} -> {command} a={action}")
            pass

        self._step += 1
        return action, command