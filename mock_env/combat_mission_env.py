def _tick(self):
    # Update time counter
    self.current_time += 1

    # Update aircraft position based on autopilot or manual control
    if self.autopilot_on:
        # Autopilot behavior: move toward enemy if detected, otherwise move left
        if self.enemy_detected:
            dx = self.enemy_pos[0] - self.aircraft_pos[0]
            dy = self.enemy_pos[1] - self.aircraft_pos[1]
            distance = math.hypot(dx, dy)

            # Maintain a reasonable distance to target
            optimal_distance = 150 if self.enemy_type == 1 else 100

            if distance > optimal_distance * 1.2:
                # Move closer but not too quickly
                speed_factor = min(1.0, (distance - optimal_distance) / 100)
                self.aircraft_vel = [dx / distance * self.aircraft_speed * speed_factor,
                                     dy / distance * self.aircraft_speed * speed_factor]
            elif distance < optimal_distance * 0.8:
                # Back away slightly
                self.aircraft_vel = [-dx / distance * self.aircraft_speed * 0.5,
                                     -dy / distance * self.aircraft_speed * 0.5]
            else:
                # Orbit around at optimal distance
                self.aircraft_vel = [dy / distance * self.aircraft_speed * 0.5,
                                     -dx / distance * self.aircraft_speed * 0.5]
        else:
            # Search pattern when no enemy detected
            self.aircraft_vel = [-2, math.sin(self.steps * 0.05) * 2]

    # Update aircraft position
    self.aircraft_pos[0] += self.aircraft_vel[0]
    self.aircraft_pos[1] += self.aircraft_vel[1]

    # Keep aircraft within screen bounds
    self.aircraft_pos[0] = max(20, min(self.screen_size[0] - 20, self.aircraft_pos[0]))
    self.aircraft_pos[1] = max(20, min(self.screen_size[1] - 20, self.aircraft_pos[1]))

    # Update radar sweep angle if in sweep mode
    if self.radar_sweep_mode == 1:
        self.radar_angle = (self.radar_angle + 3 * self.radar_sweep_direction) % 360
        if self.radar_angle >= 270 or self.radar_angle <= 90:
            self.radar_sweep_direction *= -1  # Reverse direction at limits

    # Update missiles
    for missile in self.missiles:
        # Store previous position for trail effect
        missile['prev_x'] = missile['x']
        missile['prev_y'] = missile['y']

        # Active radar missiles can seek enemy
        if missile['type'] == "active_radar" or missile['target_pos']:
            target = self.enemy_pos if missile['type'] == "active_radar" and self.enemy_alive else missile['target_pos']

            if target:
                dx = target[0] - missile['x']
                dy = target[1] - missile['y']
                distance = math.hypot(dx, dy)

                if distance > 0:
                    # Add some randomness to missile path for realism
                    jitter_x = random.uniform(-0.5, 0.5)
                    jitter_y = random.uniform(-0.5, 0.5)

                    missile['x'] += missile['speed'] * (dx / distance) + jitter_x
                    missile['y'] += missile['speed'] * (dy / distance) + jitter_y
                    missile['distance_traveled'] += missile['speed']

        # Check for missile hit
        if self.enemy_alive:
            distance_to_enemy = math.hypot(
                missile['x'] - self.enemy_pos[0],
                missile['y'] - self.enemy_pos[1]
            )

            if distance_to_enemy < 15:  # Increased hit radius
                # Calculate probability of kill based on distance traveled
                pk = missile['pk_function'](missile['distance_traveled'])

                # Debug info
                # print(f"Missile hit! Type: {missile['type']}, Distance: {missile['distance_traveled']:.1f}, PK: {pk:.2f}")

                if random.random() < pk:
                    self.enemy_alive = False
                    # Set explosion position
                    self.explosion_pos = [self.enemy_pos[0], self.enemy_pos[1]]
                    self.explosion_frame = 0
                else:
                    # Miss effect - remove missile
                    missile['distance_traveled'] = missile['range']  # Mark for removal

    # Remove missiles that are out of range or off-screen
    self.missiles = [m for m in self.missiles if
                     m['distance_traveled'] < m['range'] and
                     0 <= m['x'] <= self.screen_size[0] and
                     0 <= m['y'] <= self.screen_size[1]]

    # Update explosion animation
    if self.explosion_pos is not None:
        self.explosion_frame += 1
        if self.explosion_frame > 15:  # End of explosion animation
            self.explosion_pos = None

    # Update step counter
    self.steps += 1
    if self.radar_sweep_mode == 1:
        self.radar_angle = (self.radar_angle + 3 * self.radar_sweep_direction) % 360
        if self.radar_angle >= 270 or self.radar_angle <= 90:
            self.radar_sweep_direction *= -1  # Reverse direction at limits

    # Update missiles
    for missile in self.missiles:
        # Active radar missiles can seek enemy
        if missile['type'] == "active_radar" or missile['target_pos']:
            target = self.enemy_pos if missile['type'] == "active_radar" and self.enemy_alive else missile['target_pos']

            if target:
                dx = target[0] - missile['x']
                dy = target[1] - missile['y']
                distance = math.hypot(dx, dy)

                if distance > 0:
                    # Add some randomness to missile path for realism
                    jitter_x = random.uniform(-0.5, 0.5)
                    jitter_y = random.uniform(-0.5, 0.5)

                    missile['x'] += missile['speed'] * (dx / distance) + jitter_x
                    missile['y'] += missile['speed'] * (dy / distance) + jitter_y
                    missile['distance_traveled'] += missile['speed']

        # Check for missile hit
        if self.enemy_alive:
            distance_to_enemy = math.hypot(
                missile['x'] - self.enemy_pos[0],
                missile['y'] - self.enemy_pos[1]
            )

            if distance_to_enemy < 15:  # Increased hit radius
                # Calculate probability of kill based on distance traveled
                pk = missile['pk_function'](missile['distance_traveled'])

                # Debug info
                # print(f"Missile hit! Type: {missile['type']}, Distance: {missile['distance_traveled']:.1f}, PK: {pk:.2f}")

                if random.random() < pk:
                    self.enemy_alive = False
                else:
                    # Miss effect - create temporary visual feedback
                    pass

    # Remove missiles that are out of range or off-screen
    self.missiles = [m for m in self.missiles if
                     m['distance_traveled'] < m['range'] and
                     0 <= m['x'] <= self.screen_size[0] and
                     0 <= m['y'] <= self.screen_size[1]]

    # Update current step counter
    self.steps += 1
    import gym


from gym import spaces
import numpy as np
import random
import pygame
import math
import gym


class CombatMissionEnv(gym.Env):
    def __init__(self):
        super(CombatMissionEnv, self).__init__()

        # Environment constants
        self.max_distance = 50000  # meters
        self.max_altitude = 20000  # meters
        self.max_perception = 240  # radar perception vector size
        self.max_missiles = 4  # maximum missiles of each type
        self.radar_range = 300  # radar detection range in pixels (increased from 160)
        self.radar_sweep_angle = 120  # radar sweep angle in degrees

        # Missile properties
        self.missile_types = {
            "active_radar": {
                "range": 300,  # longer range (pixels)
                "speed": 12,  # faster
                "pk_function": lambda d: max(0.95 - (d / 250) ** 2, 0.2)
                # probability of kill function based on distance
            },
            "ir_guided": {
                "range": 180,  # shorter range (pixels)
                "speed": 10,  # slower
                "pk_function": lambda d: max(0.98 - (d / 150) ** 2, 0.3)  # higher pk at close range
            }
        }

        # Define action space
        # 0: Fire Active Radar Missile
        # 1: Fire IR Guided Missile
        # 2: Toggle Radar On/Off
        # 3: Toggle Radar Sweep Mode
        # 4: Toggle Autopilot
        # 5: Maneuver toward enemy
        # 6: Maneuver north
        # 7: Maneuver south
        # 8: Maneuver east
        # 9: Maneuver west
        # 10: Maneuver opposite of enemy
        # 11: No-op
        self.action_space = spaces.Discrete(12)

        # Define observation space
        self.observation_space = spaces.Dict({
            'distance': spaces.Box(low=0, high=self.max_distance, shape=(1,), dtype=np.float32),
            'altitude': spaces.Box(low=0, high=self.max_altitude, shape=(1,), dtype=np.float32),
            'perception': spaces.Box(low=0, high=1, shape=(self.max_perception,), dtype=np.int8),
            'radar_on': spaces.Discrete(2),
            'radar_sweep_mode': spaces.Discrete(2),
            'autopilot_on': spaces.Discrete(2),
            'active_radar_missiles': spaces.Discrete(self.max_missiles + 1),
            'ir_guided_missiles': spaces.Discrete(self.max_missiles + 1),
            'enemy_type': spaces.Discrete(3),  # 0: none, 1: aircraft, 2: ground unit
            'enemy_detected': spaces.Discrete(2)
        })

        # Rendering settings
        self.screen_size = (800, 600)
        self.aircraft_color = (0, 0, 255)  # blue
        self.ground_unit_color = (255, 0, 0)  # red
        self.dead_enemy_color = (100, 100, 100)  # gray
        self.bg_color = (255, 255, 255)  # white
        self.radar_color = (0, 255, 0, 100)  # semi-transparent green
        self.radar_sweep_color = (0, 200, 0, 150)  # more visible green
        self.active_missile_color = (255, 200, 0)  # orange-yellow for active radar missiles
        self.ir_missile_color = (255, 0, 255)  # magenta for IR missiles
        self.screen = None
        self.clock = pygame.time.Clock()
        self.font = None
        self.explosion_frames = []  # For explosion animation
        self.explosion_pos = None  # Position of current explosion
        self.explosion_frame = 0  # Current frame of explosion
        self.current_time = 0  # Step counter for animation purposes

        self.reset()

    def reset(self):
        # Aircraft state
        self.aircraft_pos = [700.0, 300.0]
        self.aircraft_alt = float(np.random.randint(1000, self.max_altitude))
        self.aircraft_vel = [0.0, 0.0]
        self.aircraft_speed = 5.0

        # Sensor state
        self.radar_on = 0
        self.radar_sweep_mode = 0  # 0: fixed forward, 1: sweeping
        self.radar_angle = 0  # current radar direction in sweep mode
        self.radar_sweep_direction = 1  # 1: clockwise, -1: counterclockwise

        # System state
        self.autopilot_on = 1
        self.active_radar_missiles = 4
        self.ir_guided_missiles = 4

        # Enemy state
        self.enemy_type = random.choice([1, 2])  # 1: aircraft, 2: ground unit
        self.enemy_alive = True
        self.enemy_detected = 0

        # Set enemy position based on type
        if self.enemy_type == 2:  # Ground unit
            self.enemy_pos = [100.0, 500.0]  # Ground units are lower
            self.enemy_alt = 0
        else:  # Aircraft
            self.enemy_pos = [100.0, float(np.random.randint(200, 400))]
            self.enemy_alt = float(np.random.randint(1000, self.max_altitude))

        # Missile state
        self.missiles = []
        self.done = False
        self.steps = 0
        self.last_detection_angle = 0  # Last angle where enemy was detected
        self.current_time = 0

        # Reset explosion animation
        self.explosion_pos = None
        self.explosion_frame = 0

        return self._get_obs()

    def _get_obs(self):
        # Calculate distance to enemy
        distance = math.hypot(self.aircraft_pos[0] - self.enemy_pos[0],
                              self.aircraft_pos[1] - self.enemy_pos[1]) if self.enemy_alive else 0

        # Initialize perception vector (all zeros)
        perception = np.zeros(self.max_perception, dtype=np.int8)

        # Only populate perception if radar is on and enemy is alive
        enemy_detected = 0
        if self.radar_on and self.enemy_alive:
            # Check if enemy is within radar range
            if distance <= self.radar_range:
                # Calculate angle to enemy (in degrees)
                dx = self.enemy_pos[0] - self.aircraft_pos[0]
                dy = self.enemy_pos[1] - self.aircraft_pos[1]
                angle_to_enemy = math.degrees(math.atan2(-dy, -dx)) % 360

                # More forgiving radar detection logic (wider detection cone)
                if self.radar_sweep_mode == 0:  # Fixed forward radar
                    # Always detect if within range for ground units (simplification)
                    if self.enemy_type == 2:
                        enemy_detected = 1
                        self.last_detection_angle = angle_to_enemy
                    else:
                        # For aircraft, still use angle detection but with wider cone
                        angle_diff = abs((angle_to_enemy - 180) % 360)
                        angle_diff = min(angle_diff, 360 - angle_diff)  # Handle wrap-around

                        if angle_diff <= self.radar_sweep_angle:  # More forgiving angle check
                            enemy_detected = 1
                            self.last_detection_angle = angle_to_enemy

                    if enemy_detected:
                        # Set perception vector (simplified representation)
                        angle_index = int((angle_to_enemy / 360) * self.max_perception)
                        perception[angle_index] = 1
                else:  # Sweeping radar
                    # Check if enemy is within current radar sweep with wider detection
                    radar_min_angle = (self.radar_angle - self.radar_sweep_angle) % 360
                    radar_max_angle = (self.radar_angle + self.radar_sweep_angle) % 360

                    # Ground units have higher radar reflectivity (easier to detect)
                    if self.enemy_type == 2:
                        # Wider detection angle for ground units
                        if radar_min_angle < radar_max_angle:
                            enemy_in_sweep = radar_min_angle <= angle_to_enemy <= radar_max_angle
                        else:  # Handling wrap-around
                            enemy_in_sweep = angle_to_enemy >= radar_min_angle or angle_to_enemy <= radar_max_angle

                        if enemy_in_sweep or distance < self.radar_range * 0.5:  # Guaranteed detection at close range
                            enemy_detected = 1
                            self.last_detection_angle = angle_to_enemy
                    else:
                        # For aircraft, use normal sweep detection but still more forgiving
                        if radar_min_angle < radar_max_angle:
                            enemy_in_sweep = radar_min_angle <= angle_to_enemy <= radar_max_angle
                        else:  # Handling wrap-around
                            enemy_in_sweep = angle_to_enemy >= radar_min_angle or angle_to_enemy <= radar_max_angle

                        if enemy_in_sweep:
                            enemy_detected = 1
                            self.last_detection_angle = angle_to_enemy

                    if enemy_detected:
                        # Set perception vector
                        angle_index = int((angle_to_enemy / 360) * self.max_perception)
                        perception[angle_index] = 1

        self.enemy_detected = enemy_detected

        return {
            'distance': np.array([distance], dtype=np.float32),
            'altitude': np.array([self.aircraft_alt], dtype=np.float32),
            'perception': perception,
            'radar_on': self.radar_on,
            'radar_sweep_mode': self.radar_sweep_mode,
            'autopilot_on': self.autopilot_on,
            'active_radar_missiles': self.active_radar_missiles,
            'ir_guided_missiles': self.ir_guided_missiles,
            'enemy_type': self.enemy_type if self.enemy_alive else 0,
            'enemy_detected': enemy_detected
        }

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, {}

        reward = -0.1  # Small negative reward for each step to encourage efficiency
        self.steps += 1

        # Process actions
        if action == 0:  # Fire Active Radar Missile
            if self.enemy_detected and self.active_radar_missiles > 0:
                self.active_radar_missiles -= 1
                self._launch_missile("active_radar")
            else:
                reward = -1  # Penalty for trying to fire when no target detected or no missiles

        elif action == 1:  # Fire IR Guided Missile
            if self.enemy_detected and self.ir_guided_missiles > 0:
                self.ir_guided_missiles -= 1
                self._launch_missile("ir_guided")
            else:
                reward = -1

        elif action == 2:  # Toggle Radar On/Off
            self.radar_on = 1 - self.radar_on

        elif action == 3:  # Toggle Radar Sweep Mode
            self.radar_sweep_mode = 1 - self.radar_sweep_mode

        elif action == 4:  # Toggle Autopilot
            self.autopilot_on = 1 - self.autopilot_on

        elif action == 5:  # Maneuver toward enemy
            if not self.autopilot_on:
                dx = self.enemy_pos[0] - self.aircraft_pos[0]
                dy = self.enemy_pos[1] - self.aircraft_pos[1]
                distance = math.hypot(dx, dy)
                if distance > 0:
                    self.aircraft_vel = [dx / distance * self.aircraft_speed,
                                         dy / distance * self.aircraft_speed]

        elif action == 6:  # Maneuver north
            if not self.autopilot_on:
                self.aircraft_vel = [0, -self.aircraft_speed]

        elif action == 7:  # Maneuver south
            if not self.autopilot_on:
                self.aircraft_vel = [0, self.aircraft_speed]

        elif action == 8:  # Maneuver east
            if not self.autopilot_on:
                self.aircraft_vel = [self.aircraft_speed, 0]

        elif action == 9:  # Maneuver west
            if not self.autopilot_on:
                self.aircraft_vel = [-self.aircraft_speed, 0]

        elif action == 10:  # Maneuver opposite of enemy
            if not self.autopilot_on:
                dx = self.enemy_pos[0] - self.aircraft_pos[0]
                dy = self.enemy_pos[1] - self.aircraft_pos[1]
                distance = math.hypot(dx, dy)
                if distance > 0:
                    self.aircraft_vel = [-dx / distance * self.aircraft_speed,
                                         -dy / distance * self.aircraft_speed]

        elif action == 11:  # No-op
            pass

        # Update simulation state
        self._tick()

        # Check if mission is complete
        if not self.enemy_alive:
            reward = 10  # Reward for destroying the enemy
            self.done = True

        # Check if aircraft crashed into enemy
        aircraft_enemy_distance = math.hypot(
            self.aircraft_pos[0] - self.enemy_pos[0],
            self.aircraft_pos[1] - self.enemy_pos[1]
        )
        if aircraft_enemy_distance < 20 and self.enemy_alive:
            reward = -10  # Big penalty for crashing
            self.done = True

        # Check if aircraft flew off-screen
        if (self.aircraft_pos[0] < 0 or self.aircraft_pos[0] > self.screen_size[0] or
                self.aircraft_pos[1] < 0 or self.aircraft_pos[1] > self.screen_size[1]):
            reward = -5  # Penalty for flying off-screen
            self.done = True

        # End episode if too many steps
        if self.steps >= 500:
            self.done = True

        return self._get_obs(), reward, self.done, {}

    def _launch_missile(self, missile_type):
        # Calculate target position (use last known position if not currently detected)
        target_pos = self.enemy_pos if self.enemy_detected else None

        if target_pos or missile_type == "active_radar":  # Active radar missiles don't need continual tracking
            missile = {
                'x': self.aircraft_pos[0],
                'y': self.aircraft_pos[1],
                'type': missile_type,
                'speed': self.missile_types[missile_type]["speed"],
                'range': self.missile_types[missile_type]["range"],
                'distance_traveled': 0,
                'pk_function': self.missile_types[missile_type]["pk_function"],
                'target_pos': target_pos.copy() if target_pos else None,
                'prev_x': self.aircraft_pos[0],  # Add previous position for trail effect
                'prev_y': self.aircraft_pos[1]
            }
            self.missiles.append(missile)

    def _tick(self):
        # Update aircraft position based on autopilot or manual control
        if self.autopilot_on:
            # Autopilot behavior: move toward enemy if detected, otherwise move left
            if self.enemy_detected:
                dx = self.enemy_pos[0] - self.aircraft_pos[0]
                dy = self.enemy_pos[1] - self.aircraft_pos[1]
                distance = math.hypot(dx, dy)
                if distance > 100:  # Keep some distance
                    self.aircraft_vel = [dx / distance * self.aircraft_speed * 0.5,
                                         dy / distance * self.aircraft_speed * 0.5]
                else:
                    self.aircraft_vel = [0, 0]  # Hold position
            else:
                self.aircraft_vel = [-2, 0]  # Move left if no enemy detected

        # Update aircraft position
        self.aircraft_pos[0] += self.aircraft_vel[0]
        self.aircraft_pos[1] += self.aircraft_vel[1]

        # Keep aircraft within screen bounds
        self.aircraft_pos[0] = max(0, min(self.screen_size[0], self.aircraft_pos[0]))
        self.aircraft_pos[1] = max(0, min(self.screen_size[1], self.aircraft_pos[1]))

        # Update radar sweep angle if in sweep mode
        if self.radar_sweep_mode == 1:
            self.radar_angle = (self.radar_angle + 3 * self.radar_sweep_direction) % 360
            if self.radar_angle >= 270 or self.radar_angle <= 90:
                self.radar_sweep_direction *= -1  # Reverse direction at limits

        # Update missiles
        for missile in self.missiles:
            # Active radar missiles can seek enemy
            if missile['type'] == "active_radar" or missile['target_pos']:
                target = self.enemy_pos if missile['type'] == "active_radar" and self.enemy_alive else missile[
                    'target_pos']

                if target:
                    dx = target[0] - missile['x']
                    dy = target[1] - missile['y']
                    distance = math.hypot(dx, dy)

                    if distance > 0:
                        missile['x'] += missile['speed'] * (dx / distance)
                        missile['y'] += missile['speed'] * (dy / distance)
                        missile['distance_traveled'] += missile['speed']

            # Check for missile hit
            if self.enemy_alive:
                distance_to_enemy = math.hypot(
                    missile['x'] - self.enemy_pos[0],
                    missile['y'] - self.enemy_pos[1]
                )

                if distance_to_enemy < 10:
                    # Calculate probability of kill based on distance traveled
                    pk = missile['pk_function'](missile['distance_traveled'])
                    if random.random() < pk:
                        self.enemy_alive = False

        # Remove missiles that are out of range or off-screen
        self.missiles = [m for m in self.missiles if
                         m['distance_traveled'] < m['range'] and
                         0 <= m['x'] <= self.screen_size[0] and
                         0 <= m['y'] <= self.screen_size[1]]

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Enhanced Combat Mission Environment")
            self.font = pygame.font.SysFont(None, 24)

        self.screen.fill(self.bg_color)

        # Draw enemy
        enemy_color = self.ground_unit_color if self.enemy_alive else self.dead_enemy_color
        if self.enemy_type == 1:  # Aircraft
            pygame.draw.polygon(self.screen, enemy_color, [
                (int(self.enemy_pos[0]), int(self.enemy_pos[1]) - 5),
                (int(self.enemy_pos[0]) - 10, int(self.enemy_pos[1]) + 5),
                (int(self.enemy_pos[0]) + 10, int(self.enemy_pos[1]) + 5)
            ])
        else:  # Ground unit
            pygame.draw.rect(self.screen, enemy_color,
                             (int(self.enemy_pos[0]) - 5, int(self.enemy_pos[1]) - 5, 10, 10))

        # Draw radar visualization if on
        if self.radar_on:
            if self.radar_sweep_mode == 0:  # Fixed forward radar
                # Create a semi-transparent surface for the radar cone
                radar_surface = pygame.Surface((self.radar_range * 2, self.radar_range * 2), pygame.SRCALPHA)
                pygame.draw.arc(
                    radar_surface, self.radar_color,
                    (0, 0, self.radar_range * 2, self.radar_range * 2),
                    np.deg2rad(180 - self.radar_sweep_angle / 2),
                    np.deg2rad(180 + self.radar_sweep_angle / 2),
                    self.radar_range
                )
                self.screen.blit(radar_surface,
                                 (int(self.aircraft_pos[0] - self.radar_range),
                                  int(self.aircraft_pos[1] - self.radar_range)))
            else:  # Sweeping radar
                # Draw radar sweep
                radar_surface = pygame.Surface((self.radar_range * 2, self.radar_range * 2), pygame.SRCALPHA)
                pygame.draw.arc(
                    radar_surface, self.radar_sweep_color,
                    (0, 0, self.radar_range * 2, self.radar_range * 2),
                    np.deg2rad(self.radar_angle - self.radar_sweep_angle / 2),
                    np.deg2rad(self.radar_angle + self.radar_sweep_angle / 2),
                    self.radar_range
                )
                self.screen.blit(radar_surface,
                                 (int(self.aircraft_pos[0] - self.radar_range),
                                  int(self.aircraft_pos[1] - self.radar_range)))

        # Draw aircraft (triangle pointing in direction of movement)
        aircraft_angle = math.degrees(math.atan2(self.aircraft_vel[1], self.aircraft_vel[0])) if not (
                    self.aircraft_vel[0] == 0 and self.aircraft_vel[1] == 0) else 180
        aircraft_points = [
            (int(self.aircraft_pos[0] + 10 * math.cos(math.radians(aircraft_angle))),
             int(self.aircraft_pos[1] + 10 * math.sin(math.radians(aircraft_angle)))),
            (int(self.aircraft_pos[0] + 5 * math.cos(math.radians(aircraft_angle + 120))),
             int(self.aircraft_pos[1] + 5 * math.sin(math.radians(aircraft_angle + 120)))),
            (int(self.aircraft_pos[0] + 5 * math.cos(math.radians(aircraft_angle - 120))),
             int(self.aircraft_pos[1] + 5 * math.sin(math.radians(aircraft_angle - 120))))
        ]
        pygame.draw.polygon(self.screen, self.aircraft_color, aircraft_points)

        # Draw missiles
        for missile in self.missiles:
            color = self.active_missile_color if missile['type'] == "active_radar" else self.ir_missile_color
            pygame.draw.circle(self.screen, color, (int(missile['x']), int(missile['y'])), 3)

        # Draw status information
        status_texts = [
            f"Radar: {'On' if self.radar_on else 'Off'}",
            f"Mode: {'Sweep' if self.radar_sweep_mode else 'Fixed'}",
            f"Autopilot: {'On' if self.autopilot_on else 'Off'}",
            f"Active Radar Missiles: {self.active_radar_missiles}",
            f"IR Guided Missiles: {self.ir_guided_missiles}",
            f"Enemy Detected: {'Yes' if self.enemy_detected else 'No'}"
        ]

        for i, text in enumerate(status_texts):
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + i * 25))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None
