import numpy as np


class PIDController:
    def __init__(self, kp, ki=0.0, kd=0.0, limits=(-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out, self.max_out = limits
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt=0.1):
        self.integral += error * dt
        self.integral = np.clip(self.integral, -0.5, 0.5)  # Küçük integral
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(output, self.min_out, self.max_out)


class RuleAgent:
    """
    STRAIGHT LINE BASIT AGENT
    - Dümdüz git
    - Lock et
    - Ateş et
    O KADAR!
    """

    def __init__(self):
        # Basit PID - sadece düzeltme için
        self.pitch_pid = PIDController(kp=0.5, ki=0.01, kd=0.03, limits=(-0.4, 0.4))
        #self.roll_pid = PIDController(kp=0.8, ki=0.01, kd=0.05, limits=(-0.6, 0.6))
        #self.yaw_pid = PIDController(kp=0.2, ki=0.005, kd=0.01, limits=(-0.2, 0.2))

        self.aligned_count = 0

    def chooseAction(self, state, dt=0.1):
        return self._compute_action(state, dt)

    def chooseActionNoNoise(self, state, dt=0.1):
        return self._compute_action(state, dt)

    def _compute_action(self, state, dt=0.1):
        # State parsing
        dx, dy, dz = state[0], state[1], state[2]  # Relative position
        roll, pitch, yaw = state[3], state[4], state[5]  # My attitude
        target_angle = state[6]  # Target angle
        locked = state[7]  # Target locked
        missile_available = state[8]  # Missile available

        # Distance
        distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) * 10000

        # Debug
        if np.random.random() < 0.01:
            print(f"Dist: {distance:.1f}m, dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}, locked={locked > 0}")

        # --- SÜPER BASIT LOGIC ---

        # Sadece hedefe doğru git
        # dy > 0 = sağda, dy < 0 = solda
        # dz > 0 = yukarıda, dz < 0 = aşağıda

        # Basit hata sinyalleri
        pitch_error = dz * 0.5  # Dikey hata
        roll_error = dy * 0.8  # Yatay hata
        yaw_error = dy * 0.1  # Yardımcı

        # PID kontrol
        pitch_cmd = self.pitch_pid.compute(pitch_error, dt)
        roll_cmd = self.roll_pid.compute(roll_error, dt)
        yaw_cmd = self.yaw_pid.compute(yaw_error, dt)

        # Ateş kontrolü
        fire_cmd = -1.0

        # Hizalanma sayacı
        if abs(dy) < 0.015 and abs(dz) < 0.015:
            self.aligned_count += 1
        else:
            self.aligned_count = 0

        # Basit ateş koşulu
        if (locked > 0 and
                missile_available > 0 and
                self.aligned_count >= 3 and
                distance < 3000):
            fire_cmd = 1.0
            print(f"🔥 FIRE! Distance: {distance:.1f}m")

        return [pitch_cmd, roll_cmd, yaw_cmd, fire_cmd]