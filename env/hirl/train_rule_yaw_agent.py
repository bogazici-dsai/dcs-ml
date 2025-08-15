import math

from environments.HarfangEnv_GYM import HarfangEnv, SimpleEnemy
import environments.dogfight_client as df
import numpy as np
import yaml
import argparse
import time
import csv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os



class DataFlowTestAgent:
    def __init__(self, log_dir="dataflow_logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "dataflow_episode_log.csv")
        self.episode_data = []
        self.episode_id = 0

    def reset(self):
        self.episode_id += 1
        self.episode_data = []

    def choose_action(self, state):
        dx, dz, dy = state[0]*10000, state[1]*10000, state[2]*10000
        plane_roll, plane_pitch, plane_yaw = state[3], state[4], state[5]
        target_angle_env = state[6]
        locked = state[7]
        missile1_state = state[8]
        oppo_euler = state[9:12]
        oppo_health = state[12]
        plane_pos = state[13:16]*10000
        oppo_pos = state[16:19]*10000
        plane_heading = state[19]

        # Distance
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        # Signed angle
        heading_rad = np.deg2rad(plane_heading)
        v_vec = np.array([np.cos(heading_rad), np.sin(heading_rad)])
        r_vec = np.array([dy, dx])
        v_norm = v_vec / (np.linalg.norm(v_vec) + 1e-8)
        r_norm = r_vec / (np.linalg.norm(r_vec) + 1e-8)
        dot = np.dot(v_norm, r_norm)
        cross = v_norm[0]*r_norm[1] - v_norm[1]*r_norm[0]
        signed_angle_rad = np.arctan2(cross, dot)
        signed_angle_deg = np.degrees(signed_angle_rad)

        # Log all step data
        step_log = {
            "episode": self.episode_id,
            "dx": dx, "dy": dy, "dz": dz,
            "distance": distance,
            "plane_heading": plane_heading,
            "plane_pitch": plane_pitch,
            "plane_roll": plane_roll,
            "signed_angle_deg": signed_angle_deg,
            "target_angle_env": target_angle_env,
            "locked": locked,
            "missile1_state": missile1_state,
            "oppo_health": oppo_health,
            "plane_pos_x": plane_pos[0],
            "plane_pos_y": plane_pos[1],
            "plane_pos_z": plane_pos[2],
            "oppo_pos_x": oppo_pos[0],
            "oppo_pos_y": oppo_pos[1],
            "oppo_pos_z": oppo_pos[2]
        }
        self.episode_data.append(step_log)
        return [0.0, 0.0, 0.0, -1.0]  # Just log, no action

    def log_episode(self, rewards, dones):
        for i, step in enumerate(self.episode_data):
            step["reward"] = rewards[i] if i < len(rewards) else None
            step["done"] = dones[i] if i < len(dones) else None

        keys = list(self.episode_data[0].keys())
        with open(self.log_path, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if f.tell() == 0:
                writer.writeheader()
            for row in self.episode_data:
                writer.writerow(row)

        print(f"\n[Episode {self.episode_id}] logged {len(self.episode_data)} steps to {self.log_path}")

# TODO: More complex fire cmd, choose which missile to shoot (DONE)
# TODO: Rule based agent and its test (Still Testing), add more commands and same(?) rule based agent for enemy
# TODO: Missile types nasil olmali hepsi meteor mi olsa? Mavi takımda TFX içn AIM SL ve Meteor olabilir eşit sayılarda, kırmızı takımda Meteor ve Mica olabilir
# TODO: General cleaning of code files etc.
# TODO: Converting current harfang gym to stable_baselines compatible, action and state spaces must be correct.
# TODO: Clean harfang system commit to repo
# TODO: start RL testing

class Agents:
    def track_cmd(self, state):
        # --- Extract values from observation ---
        dx, dy, dz = state[0]*10000, state[2]*10000, state[1]*10000
        dx_norm, dy_norm, dz_norm = state[0], state[2], state[1]
        plane_heading = state[19]             # Aircraft heading (degrees)
        altitude = state[14]*10000            # Aircraft altitude (meters)
        target_angle = state[6]
        # time.sleep(0.5)

        # --- Compute relative bearing to target (in degrees, [-180, 180]) ---
        angle_to_enemy = np.degrees(np.arctan2(dx, dy))
        relative_bearing = angle_to_enemy - plane_heading
        relative_bearing = (relative_bearing + 180) % 360 - 180

        # --- Pitch command (elevation safety check first) ---
        if altitude < 1200:
            # If altitude is very low, climb
            pitch_cmd = -0.3
        elif altitude > 8000:
            # If altitude is very high, dive
            pitch_cmd = 0.3
        else:
            # --- Aircraft pitch (normalized, then convert to degrees) ---
            plane_pitch_norm = state[3]
            plane_pitch = math.degrees(plane_pitch_norm * math.pi) * (-1)

            # --- Compute required pitch to aim at target ---
            horiz_dist = np.sqrt(dx ** 2 + dy ** 2)
            pitch_to_target = np.degrees(np.arctan2(dz, horiz_dist))

            # --- Relative pitch: required pitch minus current pitch ---
            relative_pitch = pitch_to_target - plane_pitch
            relative_pitch = (relative_pitch + 90) % 180 - 90  # Clamp to [-90, 90]

            # --- Gain scaling based on distance (closer = more aggressive) ---
            xy_dist = horiz_dist
            gain = np.interp(xy_dist, [0, 800], [1.5, 1.1])

            # --- Generate pitch command (closer = more aggressive correction) ---
            if xy_dist < 800:
                pitch_cmd = np.clip(-0.03 * relative_pitch * gain / 30, -1, 1)
            else:
                # Far: softer, proportional to dz
                pitch_cmd = np.clip(dz_norm * -0.2, -1, 1)

            # --- Extra adjustment if relative pitch is large ---
            if abs(relative_pitch) > 0.5:
                if relative_pitch > 0:
                    pitch_cmd = -0.25
                elif relative_pitch < 0:
                    pitch_cmd = 0.25

        # --- Roll command (no roll control here) ---
        roll_cmd = 0.0

        # --- Yaw command: turn nose toward the target ---
        # More aggressive for larger angles
        yaw_gain = 0.03 if abs(relative_bearing) < 10 else 0.06
        yaw_cmd = np.clip(relative_bearing * yaw_gain, -1.0, 1.0)

        distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        #print(f"pitch_to__target: {pitch_to_target:.1f}°, rel_pitch: {relative_pitch:.1f}°, xy_dist: {xy_dist:.1f}m, gain: {gain:.2f}, pitch_cmd: {pitch_cmd:.2f} plane pitch norm: {plane_pitch_norm:.2f} plane pitch: {plane_pitch:.2f} rel_bear: {relative_bearing:.1f}° |target_angle: {target_angle:.2f}| Yaw_cmd: {yaw_cmd:.2f}")
        return [pitch_cmd, roll_cmd, yaw_cmd, -1]

    def evade_cmd(self, state):
        """
        Compute evasive controls (pitch, roll, yaw, throttle) against inbound missiles.

        Inputs
        -------
        state : sequence
            Simulator observation vector. Assumes:
            - Ownship pos: state[13:16] (x, y, z) in units of 1e-4 (scaled to meters)
            - Ownship heading: state[19] in degrees
            - Ownship altitude: state[14] * 1e4 in meters (fallback to ownship y)
            - Missiles: state[21:] as dicts with keys:
                { "missile_id": str, "position": [x, y, z] } (x,y,z in meters)
              Entries with position [0,0,0] are considered non-existent.

        Outputs
        -------
        list[float, float, float, float]
            [pitch_cmd, roll_cmd, yaw_cmd, throttle_cmd]
            Commands are clamped to [-1, 1]. Throttle is kept at -1 during evasion,
            as in the original implementation.

        Tactical rationale (high level)
        -------------------------------
        - LOS (line-of-sight) and range-closure based threat selection.
        - PN-mirror style lateral acceleration demand (a_lat) as core guidance signal.
        - Three engagement phases by time-to-go (t_go):
            * Phase A (t_go > T1): go to beam (~±90° aspect) with moderate g.
            * Phase B (T2 < t_go ≤ T1): weave/jink with randomized side bias.
            * Phase C (t_go ≤ T2): hard break / last-ditch snap.
        - Altitude guard enforces minimum climb demand near the floor.
        - Lightweight IIR smoothing maintains real-time stability.

        Notes
        -----
        * This refactor preserves all numerical behavior and thresholds.
        * Internal state is maintained in self._ev (rng, low-pass memory, timing flags).
        * No interface changes: identical inputs/outputs and same class attributes used.
        """
        import numpy as np
        # ------------------------------ small utilities ------------------------------

        def clamp(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
            """Clamp a value into [lo, hi]."""
            return float(max(lo, min(hi, val)))

        def wrap_deg(angle: float) -> float:
            """Wrap angle in degrees into (-180, 180]."""
            return (angle + 180.0) % 360.0 - 180.0

        def fast_tanh(x: float) -> float:
            """Alias to emphasize use as soft limiter."""
            return float(np.tanh(x))

        def get_dt() -> float:
            """Simulator step; defaults to 0.15s if not present."""
            return getattr(self, "dt", 0.15)

        dt = max(1e-3, get_dt())  # Safety lower bound for divisions

        # ------------------------------ persistent state -----------------------------
        if not hasattr(self, "_ev"):
            rng = np.random.RandomState(1234)
            self._ev = {
                "rng": rng,
                "prev": {},  # per-missile previous { "range": ..., "az": ... }
                "lp": np.array([0.0, 0.0, 0.0]),  # low-pass (pitch, roll, yaw)
                "mid_until": 0.0,  # phase-B jink timing
                "mid_dir": None,  # phase-B jink direction
                "did_snap": False,  # phase-C one-time snap trigger
                "snap_dir": 0.0,  # phase-C chosen snap direction
            }
        S = self._ev
        rng = S["rng"]

        # ------------------------------ ownship state -------------------------------
        # Positions come scaled by 1e-4; convert to meters to match missile inputs
        agent_pos = state[13:16] * 10000.0
        heading = float(state[19])
        try:
            altitude = float(state[14] * 10000.0)
        except Exception:
            altitude = float(agent_pos[1])  # Fallback if altitude channel missing

        # ------------------------------ missile parsing -----------------------------
        # Ignore dummy missiles with [0,0,0] positions
        missiles = [m for m in state[22:] if m.get("position") != [0.0, 0.0, 0.0]]

        # If no threat exists, decay commands toward zero (keep -1 throttle as before)
        if not missiles:
            tau = 0.18
            alpha = dt / (tau + dt)
            cmd_vec = (1 - alpha) * S["lp"] + alpha * np.array([0.0, 0.0, 0.0])
            S["lp"] = cmd_vec
            return [float(cmd_vec[0]), float(cmd_vec[1]), float(cmd_vec[2]), 0.0]

        # ------------------------------ threat metrics ------------------------------
        def compute_missile_metrics(missile: dict, idx: int) -> dict:
            """
            Compute kinematics relative to ownship for a single missile.

            Returns
            -------
            dict with keys:
                id, range, closure, tgo, az, rb, dLOS
            where:
                - range : 3D slant range [m]
                - closure : range rate (positive if closing) [m/s]
                - tgo : time-to-go estimate [s]
                - az : LOS azimuth wrt world (x→east, z→north) [deg]
                - rb : relative bearing (missile az on ownship nose) in [-180,180] [deg]
                - dLOS : LOS rate [deg/s]
            """
            mid = missile.get("missile_id", f"M{idx}")
            mx, my, mz = map(float, missile["position"])

            dx = mx - agent_pos[0]
            dy = my - agent_pos[1]
            dz = mz - agent_pos[2]

            rng_3d = float(np.sqrt(dx * dx + dy * dy + dz * dz))  # total range

            # Azimuth defined in x-z plane (air combat convention in this codebase)
            az_world = np.degrees(np.arctan2(dx, dz))

            prev = S["prev"].get(mid, {"range": rng_3d, "az": az_world})
            closure = (prev["range"] - rng_3d) / dt  # positive if approaching
            dlos = wrap_deg(az_world - prev["az"]) / dt
            rbearing = wrap_deg(az_world - heading)

            # Simple t_go (avoid divide-by-zero with 1e-3)
            t_go = (rng_3d / max(closure, 1e-3)) if closure > 1e-3 else 1e9

            S["prev"][mid] = {"range": rng_3d, "az": az_world}

            return {
                "id": mid,
                "range": rng_3d,
                "closure": closure,
                "tgo": t_go,
                "az": az_world,
                "rb": rbearing,
                "dLOS": dlos,
            }

        metrics_all = [compute_missile_metrics(m, i) for i, m in enumerate(missiles)]
        ms = min(metrics_all, key=lambda k: k["tgo"])  # threat with minimal time-to-go

        # ------------------------------ PN-mirror core -------------------------------
        # Lateral acceleration demand used as proxy for evasive g
        deg2rad = np.pi / 180.0
        V_rel = max(ms["closure"], 180.0)  # min closure to keep authority
        lamdot = abs(ms["dLOS"]) * deg2rad  # LOS rate [rad/s]
        r = max(ms["range"], 200.0)  # min range for stability

        g_cap = 8.0  # g limit reflected in pitch_cmd mapping
        k1, k2 = 0.9, 0.6  # tuned as in original code
        a_lat = k1 * V_rel * lamdot + k2 * (V_rel ** 2) / (r + 1.0)
        a_lat = np.clip(a_lat, 3.0, g_cap)  # assure min evasive pull, cap at g_cap

        # ------------------------------ phase selection ------------------------------
        # Phase thresholds (unchanged)
        T1, T2 = 7.0, 3.0
        t_go = ms["tgo"]
        rb = ms["rb"]
        rb_sign = -1.0 if rb > 0.0 else 1.0  # positive rb → threat on right → turn left

        # --- helper: steer yaw toward beam aspect (±90° relative bearing) -----------
        def yaw_to_beam(rbearing: float, gain: float = 0.06, scale: float = 60.0) -> float:
            """
            Drive relative bearing toward ±90°. Returns yaw command in [-1, 1].
            gain/scale maintain original shaping; do not alter to preserve behavior.
            """
            desired = 90.0 if rbearing >= 0.0 else -90.0
            err = desired - rbearing
            err_mag = abs(err)
            dir_sign = -1.0 if rbearing > 0.0 else 1.0
            return clamp(dir_sign * fast_tanh(gain * (err_mag / scale) * 60.0))

        # --- helper: map yaw authority to bank command (keeps original feel) --------
        def bank_cmd_from_yaw(yaw_cmd: float, mag: float = 1.1) -> float:
            return clamp(fast_tanh(mag * np.sign(yaw_cmd)))

        # --- helper: translate g demand to pitch command in [-1,1] ------------------
        def g_to_pitch_cmd(g: float) -> float:
            return clamp(g / g_cap)

        yaw_cmd = roll_cmd = pitch_cmd = 0.0

        # ------------------------------ rear-aspect special --------------------------
        # When the missile is behind (>90°) and not in terminal (t_go > T2),
        # prefer a sustained beaming with slightly higher g and strong roll.
        rear_aspect = abs(rb) > 90.0 and t_go > T2
        if rear_aspect:
            desired_rb = 90.0 if rb >= 0.0 else -90.0
            err = desired_rb - rb

            yaw_cmd = clamp(fast_tanh((err / 35.0) * 1.6))
            roll_cmd = clamp(fast_tanh(np.sign(yaw_cmd) * 1.3))
            pitch_cmd = g_to_pitch_cmd(min(g_cap * 1.1, a_lat * 1.2))  # modestly higher g

            # Altitude guard inside rear-aspect branch (unchanged)
            if altitude < 1200.0:
                pitch_cmd = max(pitch_cmd, g_to_pitch_cmd(5.0))

            # Local smoothing for this branch (unchanged)
            tau = 0.05
            alpha = dt / (tau + dt)
            vec = np.array([clamp(pitch_cmd), clamp(roll_cmd), clamp(yaw_cmd)])
            smoothed = (1 - alpha) * S["lp"] + alpha * vec
            S["lp"] = smoothed
            P, R, Y = [float(clamp(c)) for c in smoothed.tolist()]
            return [P, R, Y, -1.0]

        # Reset snap bookkeeping when safely far from terminal
        if t_go > 6.0:
            S["did_snap"] = False
            S["snap_dir"] = 0.0

        # ------------------------------ phase A: far (t_go > T1) --------------------
        if t_go > T1:
            yaw_cmd = yaw_to_beam(rb, gain=0.045, scale=60.0) * 0.8
            roll_cmd = bank_cmd_from_yaw(yaw_cmd, mag=1.0) * 0.85
            pitch_cmd = g_to_pitch_cmd(min(a_lat, 4.0))

        # ------------------------------ phase B: mid (T2 < t_go ≤ T1) ---------------
        elif t_go > T2:
            now_t = getattr(self, "_t", 0.0)
            if (S["mid_dir"] is None) or (now_t >= S["mid_until"]):
                # Keep original randomization policy (35% flip vs follow rb_sign)
                S["mid_dir"] = (-rb_sign if rng.rand() < 0.35 else rb_sign)
                S["mid_until"] = now_t + rng.uniform(0.6, 0.9)

            yaw_cmd = S["mid_dir"] * abs(yaw_to_beam(rb, gain=0.06, scale=55.0))
            roll_cmd = bank_cmd_from_yaw(yaw_cmd, mag=1.15)
            pitch_cmd = g_to_pitch_cmd(min(a_lat, 6.5))

        # ------------------------------ phase C: terminal (t_go ≤ T2) ---------------
        else:
            # One-time snap away from threat side when very close
            if (t_go < 1.2) and (not S["did_snap"]):
                S["did_snap"] = True
                S["snap_dir"] = -rb_sign

            snap_dir = S["snap_dir"] if S["did_snap"] else rb_sign

            yaw_cmd = snap_dir * fast_tanh(0.14 * (min(140.0, abs(90.0 - rb)) / 40.0) * 60.0)
            roll_cmd = clamp(fast_tanh(1.4 * np.sign(yaw_cmd)))
            pitch_cmd = g_to_pitch_cmd(g_cap) * 1.2  # intentional slight over-command

        # ------------------------------ altitude guard ------------------------------
        ALT_FLOOR = 1200.0
        if altitude < ALT_FLOOR:
            # Enforce minimum pull-up demand; slightly damp roll near the floor
            pitch_cmd = max(pitch_cmd, g_to_pitch_cmd(5.0))
            roll_cmd *= 0.9

        # ------------------------------ command smoothing ---------------------------
        # Phase-aware time constant (unchanged piecewise policy)
        tau = 0.06 if t_go < 2.0 else (0.10 if t_go < 5.0 else 0.18)
        alpha = dt / (tau + dt)

        vec = np.array([clamp(pitch_cmd), clamp(roll_cmd), clamp(yaw_cmd)])
        smoothed = (1 - alpha) * S["lp"] + alpha * vec
        S["lp"] = smoothed

        P, R, Y = [float(clamp(c)) for c in smoothed.tolist()]
        return [P, R, Y, -1.0]

    def climb_cmd(self, state):
        """
        Simple climb:
        - Altitude P-control
        - Light pitch-attitude damping
        - roll = yaw = 0
        """
        import math

        # ------------- Debug controls (no behavior change) -------------
        dbg_on = bool(getattr(self, "debug_climb", False))
        dbg_every = int(getattr(self, "debug_climb_every", 10))
        dt = float(getattr(self, "dt", 0.15)) or 0.15

        if not hasattr(self, "_climb_dbg"):
            self._climb_dbg = {
                "step": 0,
                "prev_alt": None,
            }

        D = self._climb_dbg
        D["step"] += 1

        def _dbg_print(**kw):
            if not dbg_on:
                return
            if D["step"] % max(1, dbg_every) != 0:
                return
            # format a compact single-line log
            msg = "[CLIMB DBG] " + " ".join(f"{k}={v}" for k, v in kw.items())
            print(msg)

        # ---------------- Parameters (YOUR ORIGINAL LOGIC) ----------------
        target_alt_m = 4000.0
        DEAD_BAND = 200.0

        # P gains (alt_err > 0 => climb; negative command = nose-up)
        Kp_up = 0.00005
        Kp_down = 0.00012

        # Small attitude damping (deg -> command)
        Kd_att = 0.0015  # if pitch is positive (nose-up), push slightly down

        # Limits (sign: negative = nose-up)
        MAX_UP = 0.035
        MAX_DOWN = 0.080

        # ---------------- State ----------------
        altitude = float(state[14] * 10000.0)  # meters
        alt_err = target_alt_m - altitude  # + => need to climb
        pitch_deg = float(state[21]*180)  # -90..90 (as provided by your sim)

        # Estimate vertical speed for debug (does NOT affect control)
        if D["prev_alt"] is None:
            vz = 0.0
        else:
            vz = (altitude - D["prev_alt"]) / dt
        D["prev_alt"] = altitude

        # ---------------- Command (unchanged structure) ----------------
        mode = "DEADBAND" if abs(alt_err) <= DEAD_BAND else ("CLIMB" if alt_err > 0 else "DESCENT")

        # compute raw (pre-clamp) for debug
        if abs(alt_err) <= DEAD_BAND:
            pitch_raw_pre = Kd_att * pitch_deg
        else:
            if alt_err > 0:
                pitch_raw_pre = -(Kp_up * alt_err) + Kd_att * pitch_deg
            else:
                pitch_raw_pre = -(Kp_down * alt_err) + Kd_att * pitch_deg

        # Clamp (as in your original)
        if pitch_raw_pre < -MAX_UP:
            pitch_raw = -MAX_UP
            sat = "SAT_UP"  # saturated nose-up
        elif pitch_raw_pre > MAX_DOWN:
            pitch_raw = MAX_DOWN
            sat = "SAT_DOWN"  # saturated nose-down
        else:
            pitch_raw = pitch_raw_pre
            sat = "OK"

        # ---------------- Debug line ----------------
        _dbg_print(
            mode=mode,
            alt=f"{altitude:.0f}m",
            err=f"{alt_err:+.0f}m",
            vz=f"{vz:+.1f}m/s",
            pitch_deg=f"{pitch_deg:+.1f}°",
            raw=f"{pitch_raw_pre:+.4f}",
            cmd=f"{pitch_raw:+.4f}",
            limits=f"[{-MAX_UP:.3f},{MAX_DOWN:.3f}]",
            sat=sat,
            dt=f"{dt:.2f}s",
            step=D["step"],
        )
        #print(pitch_deg)
        # ---------------- Outputs (unchanged) ----------------
        roll_cmd = 0.0
        yaw_cmd = 0.0
        fire_cmd = -1.0
        return [float(pitch_raw), roll_cmd, yaw_cmd, fire_cmd]

    def fire_cmd_Meteor(self, state):
        locked = state[7]
        if locked > 0:
            fire_cmd = 0
        else:
            fire_cmd = -1.0
        return [0, 0, 0, fire_cmd]

    def fire_cmd_AIM_SL(self, state):
        locked = state[7]
        if locked > 0:
            fire_cmd = 1
        else:
            fire_cmd = -1.0
        return [0, 0, 0, fire_cmd]

    def enemys_track_cmd(self, state):
        # --- Extract values from observation ---
        dx, dy, dz = state[0] * 10000, state[2] * 10000, state[1] * 10000
        dx_norm, dy_norm, dz_norm = state[0], state[2], state[1]
        plane_heading = state[19]  # Aircraft heading (degrees)
        altitude = state[14] * 10000  # Aircraft altitude (meters)
        locked = state[7]
        target_angle = state[6]
        # time.sleep(0.5)

        # --- Compute relative bearing to target (in degrees, [-180, 180]) ---
        angle_to_enemy = np.degrees(np.arctan2(dx, dy))
        relative_bearing = angle_to_enemy - plane_heading
        relative_bearing = (relative_bearing + 180) % 360 - 180

        # --- Pitch command (elevation safety check first) ---
        if altitude < 1200:
            # If altitude is very low, climb
            pitch_cmd = -0.3
        elif altitude > 8000:
            # If altitude is very high, dive
            pitch_cmd = 0.3
        else:
            # --- Aircraft pitch (normalized, then convert to degrees) ---
            plane_pitch_norm = state[3]
            plane_pitch = math.degrees(plane_pitch_norm * math.pi) * (-1)

            # --- Compute required pitch to aim at target ---
            horiz_dist = np.sqrt(dx ** 2 + dy ** 2)
            pitch_to_target = np.degrees(np.arctan2(dz, horiz_dist))

            # --- Relative pitch: required pitch minus current pitch ---
            relative_pitch = pitch_to_target - plane_pitch
            relative_pitch = (relative_pitch + 90) % 180 - 90  # Clamp to [-90, 90]

            # --- Gain scaling based on distance (closer = more aggressive) ---
            xy_dist = horiz_dist
            gain = np.interp(xy_dist, [0, 800], [1.5, 1.1])

            # --- Generate pitch command (closer = more aggressive correction) ---
            if xy_dist < 800:
                pitch_cmd = np.clip(-0.03 * relative_pitch * gain / 30, -1, 1)
            else:
                # Far: softer, proportional to dz
                pitch_cmd = np.clip(dz_norm * -0.2, -1, 1)

            # --- Extra adjustment if relative pitch is large ---
            if abs(relative_pitch) > 0.5:
                if relative_pitch > 0:
                    pitch_cmd = -0.25
                elif relative_pitch < 0:
                    pitch_cmd = 0.25

        # --- Roll command (no roll control here) ---
        roll_cmd = 0.0

        # --- Yaw command: turn nose toward the target ---
        # More aggressive for larger angles
        yaw_gain = 0.03 if abs(relative_bearing) < 10 else 0.06
        yaw_cmd = np.clip(relative_bearing * yaw_gain, -1.0, 1.0)

        distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if locked > 0:
            fire_cmd = 1.0

        else:
            fire_cmd = -1.0

        #print(f"pitch_to__target: {pitch_to_target:.1f}°, rel_pitch: {relative_pitch:.1f}°, xy_dist: {xy_dist:.1f}m, gain: {gain:.2f}, pitch_cmd: {pitch_cmd:.2f} plane pitch norm: {plane_pitch_norm:.2f} plane pitch: {plane_pitch:.2f} rel_bear: {relative_bearing:.1f}° |target_angle: {target_angle:.2f}| Yaw_cmd: {yaw_cmd:.2f}")
        return [pitch_cmd, roll_cmd, yaw_cmd, fire_cmd]


#agents = Agents()
#state = []
#track_x1=10
#cmds = [
#    agents.track_cmd(state, x1=track_x1),
#    agents.evade_cmd(state),
#    agents.climb_cmd(state),
#    agents.fire_cmd(state)
#]
def main(args):
    import os

    current_path = os.getcwd()
    print(current_path)
    with open('env/local_config.yaml', 'r') as file:
        local_config = yaml.safe_load(file)
    if local_config['network']['ip'] == 'YOUR_IP_ADDRESS':
        raise ValueError("Please update 'network.ip' in local_config.yaml")

    df.connect(local_config["network"]["ip"], args.port)
    df.disable_log()
    df.set_renderless_mode(not args.render)
    df.set_client_update_mode(True)

    # Environment selection
    if args.env == "simple_enemy":
        env = SimpleEnemy()
    else:
        raise ValueError("Unknown env_type")

    # --- Agent selection ---
    if args.agent == "data_test":
        agent = DataFlowTestAgent()
    elif args.agent == "agents":
        agent = Agents()
    else:
        raise ValueError("Unknown agent type")

    scores, successes, evade_successes = [], [], []

    os.makedirs("trajectories", exist_ok=True)

    for ep in range(args.episodes):
        state, _ = env.reset()
        _, oppo_state = env.reset()
        done = False
        total_reward, steps = 0, 0
        max_steps = 5000

        agent_positions = []
        oppo_positions = []
        rewards = []
        dones = []

        while steps < max_steps and not done:
            agent_pos = state[13:16] * 10000
            oppo_pos = state[16:19] * 10000
            agent_positions.append(agent_pos)
            oppo_positions.append(oppo_pos)

            dx, dy, dz = state[0] * 10000, state[2] * 10000, state[1] * 10000
            distance_to_enemy = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            # RULE BASED AGENT FOR ALLY
            if steps < 300:
                args.command = "track"
            elif steps > 300:
                altitude = state[14] * 10000
                if altitude < 1000:
                    args.command = "climb"
                else:
                    args.command = "track"
                    if distance_to_enemy < 5000:
                        if distance_to_enemy < 1000 and state[7] == True:
                            args.command = "fire_aim_sl"
                        elif distance_to_enemy > 1000 and state[7] == True:
                            args.command = "fire_meteor"
            else:
                threat_detected = any(
                    m.get("position") != [0.0, 0.0, 0.0] for m in state[22:]
                )
                if threat_detected and distance_to_enemy < 8000:
                    args.command = "evade"
                else:
                    args.command = "track"

            # --- Ally action ---
            if args.command == "track":
                action = agent.track_cmd(state)
            elif args.command == "evade":
                action = agent.evade_cmd(state)
            elif args.command == "climb":
                action = agent.climb_cmd(state)
            elif args.command == "fire_meteor":
                action = agent.fire_cmd_Meteor(state)
            elif args.command == "fire_aim_sl":
                action = agent.fire_cmd_AIM_SL(state)

            # --- Enemy action (aynı komut kendi state'ine uygulanıyor) ---
            if args.command == "track":
                oppo_action = agent.enemys_track_cmd(oppo_state)
            elif args.command == "evade":
                oppo_action = agent.evade_cmd(oppo_state)
            elif args.command == "climb":
                oppo_action = agent.climb_cmd(oppo_state)
            elif args.command == "fire_meteor":
                oppo_action = agent.fire_cmd_Meteor(oppo_state)
            elif args.command == "fire_aim_sl":
                oppo_action = agent.fire_cmd_AIM_SL(oppo_state)

            # NOT: alttaki override satırı kaldırıldı
            # oppo_action = agent.enemys_track_cmd(oppo_state)

            out = env.step(action, oppo_action)
            n_state, reward, done, n_state_oppo = out[:4]
            state = n_state
            oppo_state = n_state_oppo

            total_reward += reward
            rewards.append(reward)
            dones.append(done)
            steps += 1

        if state[20] < 1.0:
            evade_result = f"VURULDU ve Ally Health: {state[20]}"
            evade_success = False
        else:
            evade_result = f"VURULMADI ve Ally Health: {state[20]}"
            evade_success = True

        success = int(getattr(env, "episode_success", False))
        scores.append(total_reward)
        successes.append(success)
        evade_successes.append(evade_success)

        if hasattr(agent, "log_episode"):
            agent.log_episode(rewards, dones)

        print(f"Episode {ep + 1}/{args.episodes} | Reward: {total_reward:.1f} | Success: {success} | Steps: {steps} | Vurulma: {evade_result} ")

        # 3D Trajectory plot
        agent_positions = np.array(agent_positions)
        oppo_positions = np.array(oppo_positions)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(agent_positions[:, 0], agent_positions[:, 2], agent_positions[:, 1], label="Agent", color="blue")
        ax.plot(oppo_positions[:, 0], oppo_positions[:, 2], oppo_positions[:, 1], label="Opponent", color="red")
        ax.scatter(agent_positions[0, 0], agent_positions[0, 2], agent_positions[0, 1], color="blue", marker='o', label='Agent Start')
        ax.scatter(oppo_positions[0, 0], oppo_positions[0, 2], oppo_positions[0, 1], color="red", marker='o', label='Opponent Start')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        ax.set_title(f"3D Trajectory (Episode {ep+1})")
        plt.tight_layout()
        plt.savefig(f"trajectories/episode_{ep+1:03d}.png")
        plt.close()

    avg_reward = np.mean(scores)
    success_rate = np.mean(successes)
    evade_success_rate = np.mean(evade_successes)
    print("\n=== Rule-Based Agent Results ===")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Evade Success Rate: {evade_success_rate:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='simple_enemy', choices=['simple_enemy'])
    parser.add_argument('--agent', type=str, default='agents', choices=['agents'],
                        help="Agent type: agents")
    parser.add_argument('--port', type=int, default=50888)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    main(args)
