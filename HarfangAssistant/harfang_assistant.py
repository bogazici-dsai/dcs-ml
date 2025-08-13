import json
import math
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from HarfangAssistant.harfang_agent_prompt import render_harfang_prompt


class HarfangAssistant:
    def __init__(self, llm, verbose: bool = True, max_rate_hz: float = 10.0):
        self.llm = llm
        self.verbose = verbose
        self.max_rate_hz = max_rate_hz
        self._last_features: Optional[Dict[str, Any]] = None
        self._last_time_s: float = 0.0
        self._last_response: Tuple[float, Dict[str, Any]] = (0.0, {"critique": "init"})

    def extract_features(self, state: np.ndarray, prev_state: Optional[np.ndarray], action: np.ndarray,
                         info: Optional[Dict[str, Any]] = None,
                         lock_duration: int = 0,
                         prev_action: Optional[np.ndarray] = None) -> Dict[str, Any]:
        def safe(idx, default=0.0):
            try:
                return float(state[idx])
            except Exception:
                return float(default)

        dx = safe(0)
        dy = safe(1)
        dz = safe(2)
        plane_euler = [safe(3), safe(4), safe(5)]
        target_angle = safe(6)
        locked = int(np.sign(safe(7, -1.0)) or -1)
        missile1_state = int(np.sign(safe(8, -1.0)) or -1)
        enemy_euler = [safe(9), safe(10), safe(11)]
        enemy_health = safe(12, 1.0)
        plane_pos = [safe(13), safe(14), safe(15)]
        enemy_pos = [safe(16), safe(17), safe(18)]
        heading = safe(19)

        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if prev_state is not None:
            pdx, pdy, pdz = float(prev_state[0]), float(prev_state[1]), float(prev_state[2])
            prev_distance = math.sqrt(pdx * pdx + pdy * pdy + pdz * pdz)
            delta_distance = prev_distance - distance
        else:
            delta_distance = 0.0

        closure_rate = float(delta_distance)

        try:
            enemy_heading_deg = float(enemy_euler[1]) * 180.0
            enemy_heading_rad = math.radians(enemy_heading_deg)
            enemy_forward = np.array([math.sin(enemy_heading_rad), 0.0, math.cos(enemy_heading_rad)])
            rel_pos = np.array([dx, dy, dz])
            denom = (np.linalg.norm(rel_pos) * np.linalg.norm(enemy_forward) + 1e-8)
            aspect_angle = math.degrees(math.acos(float(np.clip(np.dot(rel_pos, enemy_forward) / denom, -1.0, 1.0))))
        except Exception:
            aspect_angle = float('nan')

        altitude = float(plane_pos[1]) if not math.isnan(plane_pos[1]) else float('nan')
        altitude_band = "unknown"
        if not math.isnan(altitude):
            if altitude < 600:
                altitude_band = "danger_low"
            elif altitude < 2000:
                altitude_band = "low"
            elif altitude <= 7000:
                altitude_band = "optimal"
            elif altitude <= 10000:
                altitude_band = "high"
            else:
                altitude_band = "danger_high"

        engagement_too_close = distance < 500
        engagement_optimal = 1000 <= distance <= 3000
        engagement_too_far = distance > 5000
        engagement_band = (
            "too_close" if engagement_too_close else
            ("optimal" if engagement_optimal else ("too_far" if engagement_too_far else "mid"))
        )

        if prev_action is not None:
            action_smoothness = float(np.linalg.norm(np.array(action[:3]) - np.array(prev_action[:3])))
        else:
            action_smoothness = 0.0

        features: Dict[str, Any] = {
            "distance": distance,
            "delta_distance": delta_distance,
            "closure_rate": closure_rate,
            "aspect_angle": aspect_angle,
            "target_angle": target_angle,
            "altitude": altitude,
            "altitude_band": altitude_band,
            "engagement_band": engagement_band,
            "engagement_optimal": engagement_optimal,
            "engagement_too_close": engagement_too_close,
            "engagement_too_far": engagement_too_far,
            "locked": locked,
            "missile1_state": missile1_state,
            "enemy_health": enemy_health,
            "plane_euler": plane_euler,
            "enemy_euler": enemy_euler,
            "heading": heading,
            "dx_dy_dz": [dx, dy, dz],
            "action": [float(action[0]), float(action[1]), float(action[2]), float(action[3])],
            "action_smoothness": action_smoothness,
            "lock_duration": int(lock_duration),
        }
        return features

    def request_shaping(self, features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        now = time.time()
        min_interval = 1.0 / max(self.max_rate_hz, 1e-3)
        if (now - self._last_time_s) < min_interval and self._last_features is not None:
            return self._last_response

        prompt = render_harfang_prompt(features)
        try:
            resp = self.llm.invoke(prompt)
            text = resp.content if hasattr(resp, 'content') else str(resp)
            data = json.loads(text)
            shaping_delta = float(data.get("shaping_delta", 0.0))
            shaping_delta = max(-0.5, min(0.5, shaping_delta))
            self._last_time_s = now
            self._last_features = features
            self._last_response = (shaping_delta, data)
            return self._last_response
        except Exception:
            self._last_time_s = now
            self._last_features = features
            self._last_response = (0.0, {"critique": "invalid_or_timeout", "shaping_delta": 0.0})
            return self._last_response


