# Mock Harfang Environment for testing when Harfang3D is not available
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple


class MockHarfangEnhancedEnv(gym.Env):
    """
    Mock Harfang environment for testing the enhanced RL-LLM system
    when Harfang3D is not available. Simulates the 25-dimensional state space
    and basic air combat dynamics.
    """
    
    def __init__(self, max_episode_steps: int = 2000):
        """
        Initialize mock Harfang environment
        
        Args:
            max_episode_steps: Maximum steps per episode
        """
        super().__init__()
        
        # Action space: [pitch, roll, yaw, fire]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Enhanced observation space - 25 dimensions to match real Harfang
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )
        
        # Episode management
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Mock combat state
        self.ego_pos = np.array([0.0, 0.0, 5000.0])  # x, y, altitude
        self.enemy_pos = np.array([10000.0, 0.0, 5000.0])  # 10km away
        self.ego_velocity = np.array([200.0, 0.0, 0.0])  # m/s
        self.enemy_velocity = np.array([-150.0, 0.0, 0.0])  # approaching
        
        # Combat parameters
        self.ego_health = 1.0
        self.enemy_health = 1.0
        self.missile_count = 4
        self.locked = False
        self.lock_duration = 0
        
        # Tactical state
        self.engagement_phase = "BVR"
        self.threat_level = 0.3
        self.energy_state = "HIGH"
        
        print("[MOCK HARFANG] Mock environment initialized for testing")
        print("[MOCK HARFANG] 25-dimensional state space simulated")
    
    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the mock environment"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode state
        self.current_step = 0
        
        # Reset combat state with some randomization
        self.ego_pos = np.array([
            np.random.uniform(-1000, 1000),    # x position
            np.random.uniform(-1000, 1000),    # y position
            np.random.uniform(3000, 8000)      # altitude
        ])
        
        self.enemy_pos = np.array([
            np.random.uniform(8000, 15000),    # x position (8-15km away)
            np.random.uniform(-2000, 2000),    # y position
            np.random.uniform(3000, 8000)      # altitude
        ])
        
        # Reset other state
        self.ego_health = 1.0
        self.enemy_health = 1.0
        self.missile_count = 4
        self.locked = False
        self.lock_duration = 0
        self.threat_level = np.random.uniform(0.1, 0.5)
        
        observation = self._get_observation()
        info = self._create_info_dict()
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the mock environment
        
        Args:
            action: [pitch, roll, yaw, fire] action
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        # Apply action to ego aircraft
        self._apply_action(action)
        
        # Update enemy (simple AI)
        self._update_enemy()
        
        # Update combat state
        self._update_combat_state()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination conditions
        terminated = self.enemy_health <= 0 or self.ego_health <= 0
        truncated = self.current_step >= self.max_episode_steps
        
        # Create info dictionary
        info = self._create_info_dict()
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _apply_action(self, action: np.ndarray):
        """Apply action to ego aircraft"""
        pitch, roll, yaw, fire = action
        
        # Simple flight dynamics
        # Update velocity based on control inputs
        speed_change = pitch * 10.0  # Pitch affects speed
        turn_rate = roll * 0.1       # Roll affects turn rate
        
        # Update ego velocity
        self.ego_velocity[0] += speed_change
        self.ego_velocity[0] = np.clip(self.ego_velocity[0], 100, 400)  # Speed limits
        
        # Update position
        dt = 1.0  # 1 second time step
        self.ego_pos += self.ego_velocity * dt
        
        # Handle firing
        if fire > 0.5 and self.missile_count > 0 and self.locked:
            self.missile_count -= 1
            # Simple hit probability based on distance and lock duration
            distance = np.linalg.norm(self.enemy_pos - self.ego_pos)
            hit_prob = max(0.1, 0.9 - (distance / 15000)) * min(1.0, self.lock_duration / 10.0)
            
            if np.random.random() < hit_prob:
                self.enemy_health -= 0.5  # Damage enemy
                print(f"[MOCK] Missile hit! Enemy health: {self.enemy_health:.1f}")
    
    def _update_enemy(self):
        """Update enemy aircraft with simple AI"""
        # Simple enemy behavior - move toward ego aircraft
        to_ego = self.ego_pos - self.enemy_pos
        distance = np.linalg.norm(to_ego)
        
        if distance > 1000:  # Approach if far away
            direction = to_ego / distance
            self.enemy_pos += direction * 180.0  # Enemy speed 180 m/s
        
        # Enemy can also fire (simplified)
        if distance < 8000 and np.random.random() < 0.01:  # 1% chance per step
            self.ego_health -= 0.1
            self.threat_level = min(1.0, self.threat_level + 0.2)
    
    def _update_combat_state(self):
        """Update combat state (lock, engagement phase, etc.)"""
        distance = np.linalg.norm(self.enemy_pos - self.ego_pos)
        
        # Update lock state (simplified)
        if distance < 12000:  # Within radar range
            if not self.locked and np.random.random() < 0.1:  # 10% chance to acquire lock
                self.locked = True
                self.lock_duration = 0
                print(f"[MOCK] Target locked at {distance/1000:.1f}km")
        
        if self.locked:
            self.lock_duration += 1
            # Can lose lock
            if np.random.random() < 0.02:  # 2% chance to lose lock
                self.locked = False
                self.lock_duration = 0
                print(f"[MOCK] Lock lost")
        
        # Update engagement phase
        if distance > 15000:
            self.engagement_phase = "BVR"
        elif distance > 8000:
            self.engagement_phase = "INTERMEDIATE"
        elif distance > 3000:
            self.engagement_phase = "MERGE"
        elif distance > 1500:
            self.engagement_phase = "WVR"
        else:
            self.engagement_phase = "KNIFE_FIGHT"
        
        # Update threat level based on distance and enemy behavior
        base_threat = max(0.1, 1.0 - (distance / 20000))
        self.threat_level = 0.7 * self.threat_level + 0.3 * base_threat  # Smooth update
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward for the current step"""
        reward = 0.0
        
        # Base survival reward
        reward += 0.1
        
        # Distance-based reward (prefer optimal engagement range)
        distance = np.linalg.norm(self.enemy_pos - self.ego_pos)
        if 4000 <= distance <= 8000:  # Optimal engagement range
            reward += 0.5
        elif distance < 2000:  # Too close - dangerous
            reward -= 0.3
        
        # Lock maintenance reward
        if self.locked:
            reward += 0.2
            if self.lock_duration > 5:
                reward += 0.1  # Bonus for maintaining lock
        
        # Firing reward
        if action[3] > 0.5:  # Fire action
            if self.locked and self.missile_count > 0:
                reward += 1.0  # Good shot opportunity
            else:
                reward -= 0.5  # Bad shot (no lock or no missiles)
        
        # Victory/defeat rewards
        if self.enemy_health <= 0:
            reward += 100.0  # Victory
        if self.ego_health <= 0:
            reward -= 100.0  # Defeat
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get 25-dimensional observation vector"""
        
        # Calculate relative position
        rel_pos = self.enemy_pos - self.ego_pos
        distance = np.linalg.norm(rel_pos)
        
        # Basic geometric features (0-12) - compatible with original
        dx, dy, dz = rel_pos / 10000.0  # Normalized relative position
        
        # Mock aircraft attitudes
        ego_euler = [0.0, 0.0, 0.0]  # pitch, roll, yaw
        enemy_euler = [0.0, 0.0, 0.0]
        
        # Target angle (simplified)
        target_angle = np.arctan2(dy, dx) / np.pi  # Normalized
        
        # Lock and missile state
        locked_state = 1.0 if self.locked else -1.0
        missile_state = 1.0 if self.missile_count > 0 else -1.0
        
        # Enhanced tactical features (13-24)
        closure_rate = np.dot(self.ego_velocity - self.enemy_velocity, rel_pos) / distance / 1000.0
        aspect_angle = np.random.uniform(-1.0, 1.0)  # Mock aspect angle
        g_force = np.random.uniform(0.5, 2.0)  # Mock G-force
        turn_rate = np.random.uniform(-0.5, 0.5)  # Mock turn rate
        climb_rate = np.random.uniform(-0.3, 0.3)  # Mock climb rate
        
        # Threat and engagement state
        norm_lock_duration = min(1.0, self.lock_duration / 20.0)
        time_since_lock = 0.0 if self.locked else np.random.uniform(0, 1.0)
        
        # Energy and engagement flags
        high_energy_flag = 1.0 if self.energy_state == "HIGH" else 0.0
        low_energy_flag = 1.0 if self.energy_state == "LOW" else 0.0
        wvr_engagement = 1.0 if self.engagement_phase in ["WVR", "KNIFE_FIGHT"] else 0.0
        bvr_engagement = 1.0 if self.engagement_phase == "BVR" else 0.0
        
        # Construct 25-dimensional state vector
        state = np.array([
            # Basic features (0-12)
            dx, dy, dz,                           # Relative position
            ego_euler[0], ego_euler[1], ego_euler[2],  # Ego attitude
            target_angle,                         # Target angle
            locked_state,                         # Lock state
            missile_state,                        # Missile availability
            enemy_euler[0], enemy_euler[1], enemy_euler[2],  # Enemy attitude
            self.enemy_health,                    # Enemy health
            
            # Enhanced features (13-24)
            closure_rate,                         # Closure rate
            aspect_angle,                         # Aspect angle
            g_force,                             # G-force
            turn_rate,                           # Turn rate
            climb_rate,                          # Climb rate
            self.threat_level,                   # Threat level
            norm_lock_duration,                  # Normalized lock duration
            time_since_lock,                     # Time since lock
            high_energy_flag,                    # High energy flag
            low_energy_flag,                     # Low energy flag
            wvr_engagement,                      # WVR engagement flag
            bvr_engagement                       # BVR engagement flag
        ], dtype=np.float32)
        
        return state
    
    def _create_info_dict(self) -> Dict[str, Any]:
        """Create info dictionary with tactical information"""
        distance = np.linalg.norm(self.enemy_pos - self.ego_pos)
        
        return {
            'distance': distance,
            'locked': self.locked,
            'lock_duration': self.lock_duration,
            'threat_level': self.threat_level,
            'energy_state': self.energy_state,
            'engagement_phase': self.engagement_phase,
            'missile_count': self.missile_count,
            'enemy_health': self.enemy_health,
            'ego_health': self.ego_health,
            'step': self.current_step,
            'success': self.enemy_health <= 0,
            'step_success': 1 if self.enemy_health <= 0 else 0
        }
    
    def close(self):
        """Close the environment"""
        pass
    
    def render(self, mode: str = 'human'):
        """Render the environment (mock)"""
        if self.current_step % 100 == 0:  # Print every 100 steps
            distance = np.linalg.norm(self.enemy_pos - self.ego_pos)
            print(f"[MOCK RENDER] Step {self.current_step}: Distance {distance/1000:.1f}km, "
                  f"Locked: {self.locked}, Phase: {self.engagement_phase}")


# Compatibility function to create mock environment when real one not available
def create_mock_harfang_env(max_episode_steps: int = 2000):
    """Create mock Harfang environment for testing"""
    return MockHarfangEnhancedEnv(max_episode_steps=max_episode_steps)
