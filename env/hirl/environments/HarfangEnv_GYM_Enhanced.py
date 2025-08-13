# HarfangEnv_GYM_Enhanced.py - Comprehensive tactical environment for RL-LLM integration
import numpy as np
import gym
import random
import os
import inspect
import math
import time
import hirl.environments.dogfight_client as df
from hirl.environments.constants import *


class HarfangEnhancedEnv(gym.Env):
    """Enhanced Harfang environment providing comprehensive tactical information for LLM guidance"""
    
    def __init__(self, max_episode_steps=None):
        super().__init__()
        self.done = False
        self.loc_diff = 0
        
        # Action space: [pitch, roll, yaw, fire]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Enhanced observation space - now much larger to include tactical info
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32  # Expanded from 13
        )
        
        # Aircraft IDs
        self.Plane_ID_oppo = "ennemy_2"
        self.Plane_ID_ally = "ally_1"
        
        # Basic state variables
        self.Aircraft_Loc = None
        self.Oppo_Loc = None
        self.Ally_target_locked = False
        self.n_Ally_target_locked = False
        self.reward = 0
        self.Plane_Irtifa = 0
        self.now_missile_state = False
        self.missile1_state = True
        self.n_missile1_state = True
        self.missile = df.get_machine_missiles_list(self.Plane_ID_ally)
        self.missile1_id = self.missile[0] if len(self.missile) > 0 else None
        self.oppo_health = 0.2
        self.target_angle = None
        self.success = 0
        self.episode_success = False
        self.fire_success = False

        # Step control
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Enhanced tracking variables for tactical analysis
        self.lock_duration = 0
        self.prev_action = [0, 0, 0, 0]
        self.shots_fired = 0
        self.consecutive_locks = 0
        self.altitude_violations = 0
        self.reward_components = {}
        
        # NEW: Comprehensive tactical state tracking
        self.prev_distance = None
        self.prev_altitude = None
        self.prev_speed = None
        self.prev_oppo_speed = None
        self.closure_rate = 0.0
        self.aspect_angle = 0.0
        self.antenna_train_angle = 0.0
        self.energy_state = "UNKNOWN"
        self.engagement_phase = "BVR"  # BVR, MERGE, WVR, DEFENSIVE, OFFENSIVE
        self.threat_level = 0.0
        self.g_force = 0.0
        self.turn_rate = 0.0
        self.climb_rate = 0.0
        
        # Missile tracking
        self.missile_distance = float('inf')
        self.missile_time_to_impact = float('inf')
        self.missile_guidance_quality = 0.0
        
        # Performance metrics
        self.shots_on_target = 0
        self.shots_total = 0
        self.time_in_envelope = 0
        self.evasive_maneuvers = 0
        
        # Tactical timers
        self.time_since_last_lock = 0
        self.time_in_merge = 0
        self.time_defensive = 0

    def reset(self):
        """Reset environment with enhanced state tracking"""
        self._reset_basic_state()
        self._reset_tactical_state()
        self._reset_machine()
        self._reset_missile()
        
        state_ally = self._get_observation()
        df.set_target_id(self.Plane_ID_ally, self.Plane_ID_oppo)
        self.state = state_ally
        
        return np.array(state_ally, dtype=np.float32)

    def random_reset(self):
        """Reset with randomized initial positions"""
        self._reset_basic_state()
        self._reset_tactical_state()
        self._random_reset_machine()
        self._reset_missile()
        
        state_ally = self._get_observation()
        df.set_target_id(self.Plane_ID_ally, self.Plane_ID_oppo)
        self.state = state_ally
        
        return np.array(state_ally, dtype=np.float32)

    def _reset_basic_state(self):
        """Reset basic environment state"""
        self.Ally_target_locked = False
        self.n_Ally_target_locked = False
        self.missile1_state = True
        self.n_missile1_state = True
        self.success = 0
        self.done = False
        self.episode_success = False
        self.fire_success = False
        self.current_step = 0
        self.lock_duration = 0
        self.prev_action = [0, 0, 0, 0]
        self.shots_fired = 0
        self.consecutive_locks = 0
        self.altitude_violations = 0
        self.reward_components = {}

    def _reset_tactical_state(self):
        """Reset enhanced tactical tracking state"""
        self.prev_distance = None
        self.prev_altitude = None
        self.prev_speed = None
        self.prev_oppo_speed = None
        self.closure_rate = 0.0
        self.aspect_angle = 0.0
        self.antenna_train_angle = 0.0
        self.energy_state = "UNKNOWN"
        self.engagement_phase = "BVR"
        self.threat_level = 0.0
        self.g_force = 0.0
        self.turn_rate = 0.0
        self.climb_rate = 0.0
        self.missile_distance = float('inf')
        self.missile_time_to_impact = float('inf')
        self.missile_guidance_quality = 0.0
        self.shots_on_target = 0
        self.shots_total = 0
        self.time_in_envelope = 0
        self.evasive_maneuvers = 0
        self.time_since_last_lock = 0
        self.time_in_merge = 0
        self.time_defensive = 0

    def step(self, action):
        """Enhanced step function with comprehensive tactical analysis"""
        self.current_step += 1
        
        # Store previous state for delta calculations
        prev_state = self._get_current_tactical_state()
        
        # Apply action and get new state
        self._apply_action(action)
        n_state = self._get_observation()
        
        # Update tactical analysis
        self._update_tactical_analysis(prev_state, action)
        
        # Calculate reward
        self._get_reward(self.state, action, n_state)
        self.state = n_state
        
        # Check termination
        self._get_termination()
        
        # Check max step limit
        if self.max_episode_steps is not None and self.current_step >= self.max_episode_steps:
            self.done = True

        # Enhanced info dictionary with comprehensive tactical data
        info = self._build_enhanced_info()
        
        return np.array(n_state, dtype=np.float32), float(self.reward), bool(self.done), info

    def _get_current_tactical_state(self):
        """Get current tactical state for delta calculations"""
        try:
            ally_state = df.get_plane_state(self.Plane_ID_ally)
            oppo_state = df.get_plane_state(self.Plane_ID_oppo)
            
            return {
                'ally_pos': ally_state["position"],
                'ally_euler': ally_state["Euler_angles"],
                'ally_speed': ally_state.get("linear_speed", 0),
                'oppo_pos': oppo_state["position"],
                'oppo_euler': oppo_state["Euler_angles"],
                'oppo_speed': oppo_state.get("linear_speed", 0),
                'altitude': ally_state["position"][1],
                'distance': self._calculate_distance(ally_state["position"], oppo_state["position"])
            }
        except:
            return {}

    def _update_tactical_analysis(self, prev_state, action):
        """Update comprehensive tactical analysis"""
        if not prev_state:
            return
            
        try:
            current_state = self._get_current_tactical_state()
            
            # Update closure rate
            if self.prev_distance is not None:
                self.closure_rate = self.prev_distance - current_state['distance']
            self.prev_distance = current_state['distance']
            
            # Update climb rate
            if self.prev_altitude is not None:
                self.climb_rate = current_state['altitude'] - self.prev_altitude
            self.prev_altitude = current_state['altitude']
            
            # Calculate aspect angle (relative bearing)
            self.aspect_angle = self._calculate_aspect_angle(
                current_state['ally_pos'], 
                current_state['oppo_pos'],
                current_state['ally_euler'],
                current_state['oppo_euler']
            )
            
            # Estimate G-force from action magnitude
            self.g_force = np.sqrt(action[0]**2 + action[1]**2 + action[2]**2) * 9.0  # Rough estimate
            
            # Calculate turn rate from yaw/roll changes
            if len(self.prev_action) >= 3:
                self.turn_rate = abs(action[2] - self.prev_action[2]) * 180  # Convert to degrees/step
            
            # Update engagement phase
            self._update_engagement_phase(current_state['distance'])
            
            # Update energy state
            self._update_energy_state(current_state['altitude'], current_state['ally_speed'])
            
            # Update threat assessment
            self._update_threat_assessment(current_state)
            
            # Track lock duration
            if self.n_Ally_target_locked:
                self.lock_duration += 1
                self.time_since_last_lock = 0
            else:
                self.lock_duration = 0
                self.time_since_last_lock += 1
            
            # Update missile tracking if fired
            if self.now_missile_state:
                self._update_missile_tracking()
                
        except Exception as e:
            # Graceful degradation if tactical analysis fails
            pass

    def _calculate_distance(self, pos1, pos2):
        """Calculate 3D distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)

    def _calculate_aspect_angle(self, ally_pos, oppo_pos, ally_euler, oppo_euler):
        """Calculate aspect angle (relative bearing)"""
        try:
            # Vector from ally to opponent
            dx = oppo_pos[0] - ally_pos[0]
            dz = oppo_pos[2] - ally_pos[2]
            
            # Ally's heading
            ally_heading = ally_euler[1]  # Yaw angle
            
            # Angle to target
            angle_to_target = math.atan2(dx, dz)
            
            # Relative bearing
            relative_bearing = angle_to_target - ally_heading
            
            # Normalize to [-180, 180]
            while relative_bearing > math.pi:
                relative_bearing -= 2 * math.pi
            while relative_bearing < -math.pi:
                relative_bearing += 2 * math.pi
                
            return math.degrees(relative_bearing)
        except:
            return 0.0

    def _update_engagement_phase(self, distance):
        """Update tactical engagement phase based on air combat ranges"""
        if distance > 15000:
            self.engagement_phase = "BVR"         # Beyond Visual Range - long-range missiles
        elif distance > 8000:
            self.engagement_phase = "INTERMEDIATE" # Medium-range missile engagement
        elif distance > 3000:
            self.engagement_phase = "MERGE"       # Merge phase - closing to visual
        elif distance > 1500:
            self.engagement_phase = "WVR"         # Within Visual Range - short missiles/guns
        elif distance > 0:
            self.engagement_phase = "KNIFE_FIGHT" # Extremely close, guns only
        
        # Modify based on tactical situation
        if self.closure_rate < -50:  # Opponent gaining fast
            self.engagement_phase = "DEFENSIVE"
        elif self.closure_rate > 100 and distance < 5000:  # We're gaining fast
            self.engagement_phase = "OFFENSIVE"

    def _update_energy_state(self, altitude, speed):
        """Update energy state assessment"""
        try:
            # Simple energy state based on altitude and speed
            if altitude > 6000 and speed > 280:
                self.energy_state = "HIGH"
            elif altitude > 4000 and speed > 200:
                self.energy_state = "MEDIUM"
            else:
                self.energy_state = "LOW"
        except:
            self.energy_state = "UNKNOWN"

    def _update_threat_assessment(self, current_state):
        """Update threat level assessment"""
        try:
            threat = 0.0
            
            # Distance factor (closer = higher threat)
            if current_state['distance'] < 1000:
                threat += 0.4
            elif current_state['distance'] < 2000:
                threat += 0.2
            
            # Lock factor
            if self.n_Ally_target_locked:
                threat += 0.3
            
            # Energy disadvantage
            if self.energy_state == "LOW":
                threat += 0.2
            
            # Missile factor
            if self.missile_distance < 5000:
                threat += 0.3
                
            self.threat_level = min(threat, 1.0)
        except:
            self.threat_level = 0.0

    def _update_missile_tracking(self):
        """Update missile tracking information"""
        try:
            # This would need specific missile tracking API calls
            # For now, provide estimates
            self.missile_distance = self.loc_diff  # Approximate
            self.missile_time_to_impact = max(0, self.missile_distance / 300)  # Rough estimate
            self.missile_guidance_quality = 1.0 if self.n_Ally_target_locked else 0.3
        except:
            pass

    def _build_enhanced_info(self):
        """Build comprehensive info dictionary for logging and LLM analysis"""
        return {
            # Basic info
            'success': int(self.episode_success),
            'fire_success': int(self.fire_success),
            'step_success': self.success,
            'distance': self.loc_diff,
            'altitude': self.Plane_Irtifa,
            'target_locked': int(self.n_Ally_target_locked),
            'reward_components': self.reward_components.copy(),
            
            # Enhanced tactical info
            'closure_rate': self.closure_rate,
            'aspect_angle': self.aspect_angle,
            'energy_state': self.energy_state,
            'engagement_phase': self.engagement_phase,
            'threat_level': self.threat_level,
            'g_force': self.g_force,
            'turn_rate': self.turn_rate,
            'climb_rate': self.climb_rate,
            'lock_duration': self.lock_duration,
            'time_since_last_lock': self.time_since_last_lock,
            
            # Missile info
            'missile_distance': self.missile_distance,
            'missile_time_to_impact': self.missile_time_to_impact,
            'missile_guidance_quality': self.missile_guidance_quality,
            
            # Performance metrics
            'shots_fired': self.shots_fired,
            'shots_on_target': self.shots_on_target,
            'shot_accuracy': self.shots_on_target / max(1, self.shots_total),
            'time_in_envelope': self.time_in_envelope,
            'evasive_maneuvers': self.evasive_maneuvers,
            
            # Current step
            'current_step': self.current_step
        }

    def _get_observation(self):
        """Get enhanced observation with comprehensive tactical information"""
        try:
            # Get basic plane states
            Plane_state = df.get_plane_state(self.Plane_ID_ally)
            Oppo_state = df.get_plane_state(self.Plane_ID_oppo)
            
            # Normalize positions
            Plane_Pos = [Plane_state["position"][0] / NormStates["Plane_position"],
                        Plane_state["position"][1] / NormStates["Plane_position"],
                        Plane_state["position"][2] / NormStates["Plane_position"]]
            
            Oppo_Pos = [Oppo_state["position"][0] / NormStates["Plane_position"],
                       Oppo_state["position"][1] / NormStates["Plane_position"],
                       Oppo_state["position"][2] / NormStates["Plane_position"]]
            
            # Normalize Euler angles
            Plane_Euler = [Plane_state["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
                          Plane_state["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                          Plane_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]
            
            Oppo_Euler = [Oppo_state["Euler_angles"][0] / NormStates["Plane_Euler_angles"],
                         Oppo_state["Euler_angles"][1] / NormStates["Plane_Euler_angles"],
                         Oppo_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"]]
            
            # Update internal state
            self.Plane_Irtifa = Plane_state["position"][1]
            self.Aircraft_Loc = Plane_state["position"]
            self.Oppo_Loc = Oppo_state["position"]
            
            # Target lock state
            self.Ally_target_locked = self.n_Ally_target_locked
            self.n_Ally_target_locked = Plane_state["target_locked"]
            locked = 1 if self.n_Ally_target_locked else -1
            
            # Target angle
            target_angle = Plane_state['target_angle'] / 180
            self.target_angle = target_angle
            
            # Position difference
            Pos_Diff = [Plane_Pos[0] - Oppo_Pos[0], 
                       Plane_Pos[1] - Oppo_Pos[1], 
                       Plane_Pos[2] - Oppo_Pos[2]]
            
            # Health and missile state
            self.oppo_health = df.get_health(self.Plane_ID_oppo)
            oppo_hea = self.oppo_health['health_level']
            
            Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_ally)
            self.missile1_state = self.n_missile1_state
            self.n_missile1_state = Missile_state["missiles_slots"][0] if Missile_state["missiles_slots"] else False
            missile1_state = 1 if self.n_missile1_state else -1
            
            # Basic observation (first 13 elements, compatible with original)
            basic_obs = np.concatenate((
                Pos_Diff,           # [0:3] - Position difference
                Plane_Euler,        # [3:6] - Ally Euler angles
                [target_angle],     # [6] - Target angle
                [locked],           # [7] - Target locked
                [missile1_state],   # [8] - Missile available
                Oppo_Euler,         # [9:12] - Enemy Euler angles
                [oppo_hea]          # [12] - Enemy health
            ), axis=None)
            
            # Enhanced observations (additional 12 elements)
            enhanced_obs = np.array([
                self.closure_rate / 1000.0,           # [13] - Normalized closure rate
                self.aspect_angle / 180.0,            # [14] - Normalized aspect angle
                self.g_force / 9.0,                   # [15] - Normalized G-force
                self.turn_rate / 180.0,               # [16] - Normalized turn rate
                self.climb_rate / 100.0,              # [17] - Normalized climb rate
                self.threat_level,                    # [18] - Threat level [0-1]
                self.lock_duration / 100.0,          # [19] - Normalized lock duration
                self.time_since_last_lock / 100.0,   # [20] - Normalized time since lock
                1.0 if self.energy_state == "HIGH" else 0.0,  # [21] - High energy flag
                1.0 if self.energy_state == "LOW" else 0.0,   # [22] - Low energy flag
                1.0 if self.engagement_phase == "WVR" else 0.0, # [23] - WVR engagement
                1.0 if self.engagement_phase == "BVR" else 0.0  # [24] - BVR engagement
            ])
            
            # Combine basic and enhanced observations
            full_observation = np.concatenate((basic_obs, enhanced_obs), axis=None)
            
            return full_observation.astype(np.float32)
            
        except Exception as e:
            # Fallback to basic observation if enhanced fails
            return np.zeros(25, dtype=np.float32)

    # Include all the other methods from the original v2 environment
    def _get_reward(self, state, action, n_state):
        """Enhanced reward function - same as v2 but with additional tactical considerations"""
        self.reward = 0.0
        self.success = 0
        self.reward_components = {}
        self._get_loc_diff()

        # 1. Distance shaping
        dx, dy, dz = n_state[0], n_state[1], n_state[2]
        prev_dx, prev_dy, prev_dz = state[0], state[1], state[2]
        prev_distance = np.sqrt(prev_dx ** 2 + prev_dy ** 2 + prev_dz ** 2)
        curr_distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        distance_change = prev_distance - curr_distance

        distance_reward = 1.5 * np.clip(distance_change, -1, 1)
        self.reward += distance_reward
        self.reward_components['distance_change'] = distance_reward

        # 2. Distance penalty
        distance_penalty = -0.001 * min(self.loc_diff, 5000)
        self.reward += distance_penalty
        self.reward_components['distance_penalty'] = distance_penalty

        # 3. Target angle reward
        target_angle_reward = 1.5 * (1.0 - self.target_angle)
        self.reward += target_angle_reward
        self.reward_components['target_angle'] = target_angle_reward

        # 4. Lock-on bonus
        if self.n_Ally_target_locked:
            self.consecutive_locks += 1
            lock_bonus = min(0.5 * self.consecutive_locks, 2.0)
            self.reward += lock_bonus
            self.reward_components['lock_bonus'] = lock_bonus
        else:
            self.consecutive_locks = 0
            self.reward_components['lock_bonus'] = 0.0

        # 5. Altitude zone shaping
        altitude_reward = 0.0
        if self.Plane_Irtifa < 2000:
            altitude_reward = -5.0 * (2000 - self.Plane_Irtifa) / 1000
        elif self.Plane_Irtifa > 7000:
            altitude_reward = -5.0 * (self.Plane_Irtifa - 7000) / 1000

        if self.Plane_Irtifa < 600:
            altitude_reward = -50.0
            self.altitude_violations += 1
        elif self.Plane_Irtifa > 10000:
            altitude_reward = -50.0
            self.altitude_violations += 1

        self.reward += altitude_reward
        self.reward_components['altitude'] = altitude_reward

        # 6. Missile firing logic
        firing_reward = 0.0
        if self.now_missile_state:
            self.shots_fired += 1
            self.shots_total += 1
            if self.missile1_state and not self.Ally_target_locked:
                firing_reward = -5.0
                self.success = -1
            elif self.missile1_state and self.Ally_target_locked:
                firing_reward = 15.0
                self.success = 1
                self.fire_success = True
                self.shots_on_target += 1
            else:
                firing_reward = -1.0

        self.reward += firing_reward
        self.reward_components['firing'] = firing_reward

        # 7. Victory reward
        victory_reward = 0.0
        if self.oppo_health['health_level'] <= 0.1 and self.fire_success:
            victory_reward = 100.0

        self.reward += victory_reward
        self.reward_components['victory'] = victory_reward

        # 8. Time penalty
        time_penalty = -0.02
        self.reward += time_penalty
        self.reward_components['time_penalty'] = time_penalty

        # 9. Action smoothness
        if hasattr(self, 'prev_action'):
            action_change = np.sum(np.abs(np.array(action[:3]) - np.array(self.prev_action[:3])))
            smoothness_reward = -0.1 * action_change
            self.reward += smoothness_reward
            self.reward_components['smoothness'] = smoothness_reward
        else:
            self.reward_components['smoothness'] = 0.0

        self.prev_action = action.copy()

        # 10. Engagement zone reward
        engagement_reward = 0.0
        if 1000 <= self.loc_diff <= 3000:
            engagement_reward = 0.5
        elif 500 <= self.loc_diff <= 5000:
            engagement_reward = 0.2
        else:
            engagement_reward = -0.3

        self.reward += engagement_reward
        self.reward_components['engagement_range'] = engagement_reward

        # NEW: 11. Tactical positioning reward
        tactical_reward = 0.0
        if abs(self.aspect_angle) < 30:  # Good nose-on positioning
            tactical_reward += 0.2
        if self.energy_state == "HIGH":
            tactical_reward += 0.1
        
        self.reward += tactical_reward
        self.reward_components['tactical_positioning'] = tactical_reward

        self.reward_components['total'] = self.reward
        return self.reward

    # Copy all other necessary methods from the v2 environment
    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))
        df.set_plane_pitch(self.Plane_ID_oppo, float(0))
        df.set_plane_roll(self.Plane_ID_oppo, float(0))
        df.set_plane_yaw(self.Plane_ID_oppo, float(0))

        if float(action_ally[3] > 0):
            df.fire_missile(self.Plane_ID_ally, 0)
            self.now_missile_state = True
        else:
            self.now_missile_state = False

        df.update_scene()

    def _get_termination(self):
        if self.Plane_Irtifa < 400 or self.Plane_Irtifa > 12000:
            self.done = True

        if self.oppo_health['health_level'] <= 0:
            self.done = True
            self.episode_success = True

        if self.altitude_violations > 50:
            self.done = True

    def _reset_machine(self):
        df.reset_machine("ally_1")
        df.reset_machine("ennemy_2")
        df.set_health("ennemy_2", 0.2)
        self.oppo_health = 0.2
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def _random_reset_machine(self):
        df.reset_machine("ally_1")
        df.reset_machine("ennemy_2")
        df.set_health("ennemy_2", 0.2)
        self.oppo_health = 0.2
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.reset_machine_matrix(
            self.Plane_ID_ally,
            0 + random.randint(-100, 100),
            3500 + random.randint(-100, 100),
            -4000 + random.randint(-100, 100), 0, 0, 0
        )
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def _reset_missile(self):
        self.now_missile_state = False
        df.rearm_machine(self.Plane_ID_ally)

    def _get_loc_diff(self):
        self.loc_diff = (((self.Aircraft_Loc[0] - self.Oppo_Loc[0]) ** 2) +
                        ((self.Aircraft_Loc[1] - self.Oppo_Loc[1]) ** 2) +
                        ((self.Aircraft_Loc[2] - self.Oppo_Loc[2]) ** 2)) ** 0.5

    def get_pos(self):
        plane_state = df.get_plane_state(self.Plane_ID_ally)
        return np.array([plane_state["position"][0], plane_state["position"][1], plane_state["position"][2]])

    def get_oppo_pos(self):
        plane_state = df.get_plane_state(self.Plane_ID_oppo)
        return np.array([plane_state["position"][0], plane_state["position"][1], plane_state["position"][2]])

    def save_parameters_to_txt(self, log_dir):
        source_code1 = inspect.getsource(self._get_reward)
        source_code2 = inspect.getsource(self._reset_machine)
        source_code3 = inspect.getsource(self._get_termination)
        filename = os.path.join(log_dir, "enhanced_log.txt")
        with open(filename, 'w', encoding='utf-8') as file:
            file.write("ENHANCED HARFANG ENVIRONMENT WITH COMPREHENSIVE TACTICAL INFO\n")
            file.write("=" * 60 + "\n")
            file.write(source_code1)
            file.write('\n' + "=" * 60 + "\n")
            file.write(source_code2)
            file.write('\n' + "=" * 60 + "\n")
            file.write(source_code3)
