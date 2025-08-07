import math

from environments.HarfangEnv_GYM import HarfangEnv, HarfangSerpentineEnv, HarfangCircularEnv,HarfangSmoothZigzagEnemyEnv, HarfangDoctrineEnemyEnv, HarfangTacticalEnv
import environments.dogfight_client as df
import numpy as np
import yaml
import argparse
import time
import csv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class OnlyYawAgent:
    def choose_action(self, state):
        dy = state[1]
        if dy > 0.01:
            yaw_cmd = 1.0
        elif dy < -0.01:
            yaw_cmd = -1.0
        else:
            yaw_cmd = 0.0
        return [0.0, 0.0, yaw_cmd, -1.0]

class Straight_Agent:
    def choose_action(self, state):
        dx, dy, dz = state[0], state[1], state[2]
        target_angle = state[6]
        locked = state[7]
        missile1_state = state[8]
        pitch_cmd = 0.0
        yaw_cmd = 0.0
        roll_cmd = 0.0
        if (locked > 0) and (missile1_state > 0) and (abs(target_angle) < 0.07):
            fire_cmd = 1.0
        else:
            fire_cmd = -1.0
        return [pitch_cmd, yaw_cmd, roll_cmd, fire_cmd]

class CircularAgent_v1:
    def choose_action(self, state):
        dx_norm, dy_norm, dz_norm = state[0] , state[2] , state[1]
        dx, dy, dz = state[0]*10000, state[2]*10000, state[1]*10000
        plane_heading = state[19]
        target_angle = state[6]

        # 1. Angle to enemy in global frame (in degrees)
        angle_to_enemy = np.degrees(np.arctan2(dx, dy))  # returns -180 to +180
        print("angle to enemy:", angle_to_enemy)
        print("plane heading:", plane_heading)
        # 2. Relative bearing (enemy from your nose)
        relative_bearing = angle_to_enemy - plane_heading

        # 3. Normalize to [-180, 180]
        relative_bearing = (relative_bearing + 180) % 360 - 180

        print(f"Signed relative bearing: {relative_bearing:.1f} degrees")
        if relative_bearing < 0:
            print("Enemy is to the LEFT")
        elif relative_bearing > 0:
            print("Enemy is to the RIGHT")
        else:
            print("Enemy is dead ahead")

        #time.sleep(1)
        locked = state[7]
        missile1_state = state[8]

        #state[13]*10000
        #state[14]*10000
        z_=state[14]*10000

        if z_ > 3800:
            pitch_cmd = -0.0011
        else:
            pitch_cmd = np.clip(dy_norm * -0.015, -1.0, 1.0)

        yaw_cmd = np.clip(dx_norm * -0.65, -1.0, 1.0)  # Sola/sağa takip
        roll_cmd = 0.0  # Roll'a gerek yok
        #and (abs(target_angle) < 0.07)
        # Ateş et - basit koşul
        if (locked > 0) and (missile1_state > 0) :
            fire_cmd = 1.0
        else:
            fire_cmd = -1.0

        return [pitch_cmd, roll_cmd, -yaw_cmd, fire_cmd]


class CircularAgent_v2:
    """
    Takip, manevra, güvenlik ve ateş kararları ile geliştirilmiş Harfang3D agenti.
    - Sol/sağ kararında relative bearing kullanır
    - Roll, pitch, yaw'u gerçekçi şekilde koordine eder
    - Yerden aşırı alçak ya da yüksek uçuşu önler
    - Gerçekçi ateş kuralı içerir
    """

    def choose_action(self, state):
        # --- Observation'dan değerleri çek ---
        dx_norm, dy_norm, dz_norm = state[0], state[2], state[1]
        dx, dy, dz = state[0]*10000, state[2]*10000, state[1]*10000
        plane_heading = state[19]  # 0-360 derece
        target_angle = state[6]
        locked = state[7]
        missile1_state = state[8]
        altitude = state[14]*10000  # metrede
        time.sleep(0.15)
        # --- Relative bearing hesapla ---
        angle_to_enemy = np.degrees(np.arctan2(dx, dy))  # -180,+180 arası
        relative_bearing = angle_to_enemy - plane_heading
        relative_bearing = (relative_bearing + 180) % 360 - 180  # [-180,180]

        # --- Roll komutu: relative bearing ile (kanat yatırıp dön) ---
        roll_cmd = 0

        # --- Pitch komutu: yer/irtifa güvenliğiyle ---
        if altitude < 1200:
            pitch_cmd = 0.3   # Alçakta: tırman
        elif altitude > 9000:
            pitch_cmd = -0.3  # Çok yüksekte: dalış
        else:
            pitch_cmd = np.clip(dz_norm * -0.07, -0.8, 0.8)

        # --- Yaw komutu: hafif düzeltme ---
        yaw_cmd = np.clip(dx_norm * -0.6, -0.5, 0.5)

        # --- Ateşleme kararı: iyi hizalanınca ve yakınsa ---
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        if (locked > 0) and (missile1_state > 0) and (abs(relative_bearing) < 45) and (distance < 2500):
            fire_cmd = 1.0
        else:
            fire_cmd = -1.0

        # --- Debug ---
        print(f"altitude: {altitude:.0f} rel_bear: {relative_bearing:.1f} angle_to_enemy: {angle_to_enemy:.2f} dist: {distance:.0f} pitch: {pitch_cmd:.2f} roll: {roll_cmd:.2f}")

        return [pitch_cmd, 0.0, -yaw_cmd, fire_cmd]

class CircularAgent_v3:
    def __init__(self):
        self.fired = False  # Ateşlendi mi flag'i

    def reset(self):
        """Yeni episode/başlangıç için flag'i resetle."""
        self.fired = False
    def choose_action(self, state):
        dx = state[0] * 10000
        dy = state[2] * 10000
        dz = state[1] * 10000
        plane_heading = state[19]
        locked = state[7]
        missile1_state = state[8]
        altitude = state[14]*10000
        time.sleep(0.15)
        # -- İleri yön vektörü (burun yönü) --
        heading_rad = np.deg2rad(plane_heading)
        vx = np.cos(heading_rad)
        vy = np.sin(heading_rad)
        v = np.array([vx, vy])

        # -- Düşman vektörü --
        r = np.array([dy, dx])

        # -- Normalize et --
        v_norm = v / (np.linalg.norm(v) + 1e-8)
        r_norm = r / (np.linalg.norm(r) + 1e-8)

        # -- Signed angle --
        dot = np.dot(v_norm, r_norm)
        cross = v_norm[0]*r_norm[1] - v_norm[1]*r_norm[0]
        angle_rad = np.arctan2(cross, dot)  # ŞU AN: 0 tam karşı, +90 sağ, -90 sol
        # Klasik tanımda, +sağ ve -sol istersen işaret değiştir
        #angle_rad = -angle_rad

        #print(f"Signed angle (deg): {np.degrees(angle_rad):.1f}")
        #if abs(np.degrees(angle_rad)) < 5:
            #print("Enemy is dead ahead")
        #elif np.degrees(angle_rad) > 0:
            #print("Enemy is to the RIGHT")
        #else:
            #print("Enemy is to the LEFT")

        # Roll, pitch, yaw, fire komutlarını aşağıya istediğin gibi ekleyebilirsin.
        roll_cmd = np.clip(angle_rad * 1.2, -1.0, 1.0)
        # Eğer daha önce ateş ettiyse pitch sabit kalsın:

         # Normal pitch hesabı:
        if altitude < 1200:
            pitch_cmd = -0.05
        elif altitude > 4200:
            pitch_cmd = 0.01
        else:
            pitch_cmd = np.clip(dz / 10000 * -0.08, -0.8, 0.8)
        if self.fired:
            yaw_cmd = 1
        else:
            yaw_cmd = np.clip(-dx/10000 * np.sign(np.degrees(angle_rad)) * -0.7, -0.65, 0.65)

        distance = np.linalg.norm(r)
        if (locked > 0) and (missile1_state > 0) :
            fire_cmd = 1.0
            self.fired = True  # Artık ateşlendi olarak işaretle
        else:
            fire_cmd = -1.0

        #print(f"altitude: {altitude:.0f} signed_angle: {np.degrees(angle_rad):.1f} dist: {distance:.0f} roll: {roll_cmd:.2f}")

        return [pitch_cmd, 0, -yaw_cmd, fire_cmd]

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

class Agents:
    def track_cmd(self, state):
        # --- Extract values from observation ---
        dx, dy, dz = state[0]*10000, state[2]*10000, state[1]*10000
        dx_norm, dy_norm, dz_norm = state[0], state[2], state[1]
        plane_heading = state[19]             # Aircraft heading (degrees)
        altitude = state[14]*10000            # Aircraft altitude (meters)
        target_angle = state[6]

        #time.sleep(0.025)

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

        print(
            f"pitch_to__target: {pitch_to_target:.1f}°, rel_pitch: {relative_pitch:.1f}°, xy_dist: {xy_dist:.1f}m, gain: {gain:.2f}, pitch_cmd: {pitch_cmd:.2f} plane pitch norm: {plane_pitch_norm:.2f} plane pitch: {plane_pitch:.2f} rel_bear: {relative_bearing:.1f}° |target_angle: {target_angle:.2f}| Yaw_cmd: {yaw_cmd:.2f}")
        return [pitch_cmd, roll_cmd, yaw_cmd, -1]


    def evade_cmd(self, state, x1=0, y1=0, x2=0, y2=0):
        pitch_cmd, yaw_cmd, fire_cmd = 0.0, 0.0, 0.0

        return [pitch_cmd, 0, -yaw_cmd, fire_cmd]
    def climb_cmd(self, state, x1=0, y1=0, x2=0, y2=0):
        pitch_cmd, yaw_cmd, fire_cmd = 0.0, 0.0, 0.0
        return [pitch_cmd, 0, -yaw_cmd, fire_cmd]
    def fire_cmd(self, state, x1=0, y1=0, x2=0, y2=0):
        pitch_cmd, yaw_cmd, fire_cmd = 0.0, 0.0, 0.0
        return [pitch_cmd, 0, -yaw_cmd, fire_cmd]

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
    with open('local_config.yaml', 'r') as file:
        local_config = yaml.safe_load(file)
    if local_config['network']['ip'] == 'YOUR_IP_ADDRESS':
        raise ValueError("Please update 'network.ip' in local_config.yaml")

    df.connect(local_config["network"]["ip"], args.port)
    df.disable_log()
    df.set_renderless_mode(not args.render)
    df.set_client_update_mode(True)

    # Environment selection
    if args.env == "straight_line":
        env = HarfangEnv()
    elif args.env == "serpentine":
        env = HarfangSerpentineEnv()
    elif args.env == "circular":
        env = HarfangCircularEnv()
    elif args.env == "zigzag":
        env = HarfangSmoothZigzagEnemyEnv()
    elif args.env == "fight":
        env = HarfangDoctrineEnemyEnv()
    elif args.env == "fight2":
        env = HarfangTacticalEnv()
    else:
        raise ValueError("Unknown env_type")

    # --- Agent selection ---
    if args.agent == "yaw":
        agent = OnlyYawAgent()
    elif args.agent == "lead":
        agent = Straight_Agent()
    elif args.agent == "circ1":
        agent = CircularAgent_v1()
    elif args.agent == "circ2":
        agent = CircularAgent_v2()
    elif args.agent == "circ3":
        agent = CircularAgent_v3()
    elif args.agent == "data_test":
        agent = DataFlowTestAgent()
    elif args.agent == "agents":
        agent = Agents()
    else:
        raise ValueError("Unknown agent type")

    scores, successes = [], []

    # Trajectory görselleri için klasör oluştur
    os.makedirs("trajectories", exist_ok=True)

    for ep in range(args.episodes):
        state = env.reset()
        done = False
        total_reward, steps = 0, 0
        max_steps = 100000

        agent_positions = []
        oppo_positions = []

        rewards = []
        dones = []

        #agent.reset()  # Her episode başında agent'ı sıfırla

        while steps < max_steps and not done:
            agent_pos = state[13:16] * 10000
            oppo_pos = state[16:19] * 10000
            agent_positions.append(agent_pos)
            oppo_positions.append(oppo_pos)
            if args.command == "track":
                action = agent.track_cmd(state)
            elif args.agent == "evade":
                action = agent.evade_cmd(state)

            out = env.step(action)
            n_state, reward, done = out[:3]
            state = n_state
            total_reward += reward
            rewards.append(reward)
            dones.append(done)
            steps += 1

        success = int(getattr(env, "episode_success", False))
        scores.append(total_reward)
        successes.append(success)
        # Dataflow agent için log dosyasına kaydet
        if hasattr(agent, "log_episode"):
            agent.log_episode(rewards, dones)
        print(f"Episode {ep + 1}/{args.episodes} | Reward: {total_reward:.1f} | Success: {success} | Steps: {steps}")

        # ----------- 3D plot ve kaydetme bölümü -----------
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
    print("\n=== Rule-Based Agent Results ===")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='straight_line', choices=['straight_line', 'serpentine', 'circular','zigzag', 'fight', 'fight2'])
    parser.add_argument('--agent', type=str, default='yaw', choices=['yaw', 'lead', 'circ1', 'circ2','circ3','data_test', 'agents'],
                        help="Agent type: 'yaw' for OnlyYawAgent, 'lead' for LeadPursuitRuleAgent")
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--command', type=str, default='track', choices=['track','evade'])
    args = parser.parse_args()
    main(args)
