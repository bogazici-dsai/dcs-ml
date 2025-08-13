# HarfangAssistant_Enhanced.py - Comprehensive tactical assistant for dogfight RL-LLM integration
import json
import math
import time
import numpy as np
from typing import Any, Dict, Optional, Tuple


class HarfangTacticalAssistant:
    """Enhanced tactical assistant providing comprehensive dogfight guidance for RL agents"""
    
    def __init__(self, llm, verbose: bool = True, max_rate_hz: float = 10.0):
        self.llm = llm
        self.verbose = verbose
        self.max_rate_hz = max_rate_hz
        self._last_features: Optional[Dict[str, Any]] = None
        self._last_time_s: float = 0.0
        self._last_response: Tuple[float, Dict[str, Any]] = (0.0, {"critique": "init"})
        
        # Tactical knowledge base - Aircraft Combat Ranges
        self.engagement_ranges = {
            "BVR": (15000, float('inf')),     # Beyond Visual Range - long-range missile engagement
            "INTERMEDIATE": (8000, 15000),    # Intermediate range - medium-range missiles
            "MERGE": (3000, 8000),            # Merge phase - closing to visual range
            "WVR": (1500, 3000),              # Within Visual Range - short-range missiles/guns
            "KNIFE_FIGHT": (0, 1500)          # Knife fight - extremely close maneuvering combat
        }
        
        self.altitude_bands = {
            "DECK": (0, 500),              # On the deck - very dangerous, used for terrain masking
            "LOW": (500, 3000),            # Low altitude - limits radar coverage, risky
            "MEDIUM": (3000, 8000),        # Medium altitude - good maneuvering altitude
            "HIGH": (8000, 15000),         # High altitude - energy advantage, good BVR position
            "VERY_HIGH": (15000, 25000)    # Very high - maximum energy, long-range engagement
        }
        
        # Tactical state tracking
        self.engagement_history = []
        self.action_effectiveness = {}
        self.tactical_recommendations = []

    def extract_features(self, state: np.ndarray, prev_state: Optional[np.ndarray], action: np.ndarray,
                        info: Optional[Dict[str, Any]] = None,
                        lock_duration: int = 0,
                        prev_action: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Extract comprehensive tactical features from enhanced environment state"""
        
        def safe_get(idx, default=0.0):
            try:
                return float(state[idx])
            except Exception:
                return float(default)

        # Basic geometric features (first 13 elements - compatible with original)
        dx = safe_get(0)
        dy = safe_get(1) 
        dz = safe_get(2)
        plane_euler = [safe_get(3), safe_get(4), safe_get(5)]
        target_angle = safe_get(6)
        locked = int(np.sign(safe_get(7, -1.0)) or -1)
        missile1_state = int(np.sign(safe_get(8, -1.0)) or -1)
        enemy_euler = [safe_get(9), safe_get(10), safe_get(11)]
        enemy_health = safe_get(12, 1.0)

        # Enhanced tactical features (elements 13-24 from enhanced environment)
        closure_rate = safe_get(13) * 1000.0  # Denormalize
        aspect_angle = safe_get(14) * 180.0   # Denormalize
        g_force = safe_get(15) * 9.0          # Denormalize
        turn_rate = safe_get(16) * 180.0      # Denormalize
        climb_rate = safe_get(17) * 100.0     # Denormalize
        threat_level = safe_get(18)
        norm_lock_duration = safe_get(19)
        time_since_lock = safe_get(20) * 100.0
        high_energy_flag = bool(safe_get(21))
        low_energy_flag = bool(safe_get(22))
        wvr_engagement = bool(safe_get(23))
        bvr_engagement = bool(safe_get(24))

        # Calculate derived metrics
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        
        # Determine altitude and range bands
        altitude = abs(dy * 10000)  # Rough denormalization
        altitude_band = self._classify_altitude(altitude)
        engagement_band = self._classify_engagement_range(distance)
        
        # Energy state assessment
        energy_state = "HIGH" if high_energy_flag else ("LOW" if low_energy_flag else "MEDIUM")
        
        # Engagement phase
        engagement_phase = "WVR" if wvr_engagement else ("BVR" if bvr_engagement else "MERGE")
        
        # Calculate delta metrics if previous state available
        delta_distance = 0.0
        delta_altitude = 0.0
        action_smoothness = 0.0
        
        if prev_state is not None:
            try:
                prev_dx = float(prev_state[0])
                prev_dy = float(prev_state[1])
                prev_dz = float(prev_state[2])
                prev_distance = math.sqrt(prev_dx**2 + prev_dy**2 + prev_dz**2)
                delta_distance = prev_distance - distance
                delta_altitude = float(prev_state[1]) - dy
            except:
                pass
        
        if prev_action is not None:
            try:
                action_change = np.sum(np.abs(np.array(action[:3]) - np.array(prev_action[:3])))
                action_smoothness = float(action_change)
            except:
                pass

        # Tactical situation assessment
        tactical_situation = self._assess_tactical_situation(
            distance, aspect_angle, closure_rate, locked, energy_state, 
            engagement_phase, threat_level
        )
        
        # Build comprehensive feature dictionary
        features = {
            # Basic geometry
            "distance": distance,
            "delta_distance": delta_distance,
            "aspect_angle": aspect_angle,
            "closure_rate": closure_rate,
            "target_angle": target_angle,
            "altitude": altitude,
            "delta_altitude": delta_altitude,
            "altitude_band": altitude_band,
            "engagement_band": engagement_band,
            
            # Lock and weapons
            "locked": locked,
            "missile1_state": missile1_state,
            "enemy_health": enemy_health,
            "lock_duration": int(lock_duration),
            "time_since_lock": time_since_lock,
            
            # Aircraft state
            "plane_euler": plane_euler,
            "enemy_euler": enemy_euler,
            "g_force": g_force,
            "turn_rate": turn_rate,
            "climb_rate": climb_rate,
            "energy_state": energy_state,
            "engagement_phase": engagement_phase,
            
            # Tactical assessment
            "threat_level": threat_level,
            "tactical_situation": tactical_situation,
            
            # Action analysis
            "action": action.tolist() if hasattr(action, 'tolist') else list(action),
            "action_smoothness": action_smoothness,
            
            # Derived tactical metrics
            "nose_on_target": abs(target_angle) < 0.2,
            "in_firing_envelope": self._in_firing_envelope(distance, aspect_angle, locked),
            "energy_advantage": energy_state == "HIGH",
            "defensive_position": threat_level > 0.6,
            "optimal_range": self._in_optimal_range(distance, engagement_phase),
            
            # Advanced tactical features
            "pursuit_geometry": self._assess_pursuit_geometry(aspect_angle, closure_rate),
            "missile_employment_zone": self._assess_missile_zone(distance, locked, aspect_angle),
            "defensive_urgency": self._assess_defensive_urgency(threat_level, distance, closure_rate),
            "energy_management_priority": self._assess_energy_priority(energy_state, altitude, threat_level)
        }
        
        # Add info dictionary data if available
        if info:
            for key, value in info.items():
                if key not in features:  # Don't overwrite existing features
                    features[f"info_{key}"] = value
        
        return features

    def _classify_altitude(self, altitude: float) -> str:
        """Classify altitude into tactical bands"""
        for band, (min_alt, max_alt) in self.altitude_bands.items():
            if min_alt <= altitude < max_alt:
                return band
        return "UNKNOWN"

    def _classify_engagement_range(self, distance: float) -> str:
        """Classify engagement range into tactical bands"""
        for band, (min_dist, max_dist) in self.engagement_ranges.items():
            if min_dist <= distance < max_dist:
                return band
        return "UNKNOWN"

    def _assess_tactical_situation(self, distance: float, aspect_angle: float, 
                                 closure_rate: float, locked: int, energy_state: str,
                                 engagement_phase: str, threat_level: float) -> str:
        """Assess overall tactical situation in air combat context"""
        
        if threat_level > 0.8:
            return "DEFENSIVE_CRITICAL"  # Under immediate missile threat or gun tracking
        elif threat_level > 0.6:
            return "DEFENSIVE"  # Being targeted, need evasive action
        elif locked > 0 and distance < 4000 and abs(aspect_angle) < 20:
            return "FIRING_SOLUTION"  # Good missile shot opportunity
        elif locked > 0 and distance < 1500 and abs(aspect_angle) < 30:
            return "GUN_TRACKING"  # Close range gun engagement opportunity
        elif energy_state == "HIGH" and distance > 8000:
            return "BVR_ADVANTAGE"  # High altitude/speed advantage for BVR engagement
        elif engagement_phase == "BVR":
            return "BVR_POSITIONING"  # Maneuvering for long-range missile shot
        elif engagement_phase == "WVR":
            return "DOGFIGHTING"  # Close-in air combat maneuvering (ACM)
        elif engagement_phase == "KNIFE_FIGHT":
            return "GUNS_GUNS_GUNS"  # Extremely close range, guns only
        elif distance > 15000:
            return "LONG_RANGE_INTERCEPT"  # Long range approach/intercept
        else:
            return "MANEUVERING"  # General tactical maneuvering

    def _in_firing_envelope(self, distance: float, aspect_angle: float, locked: int) -> bool:
        """Determine if in optimal missile firing envelope for air combat"""
        # Modern air-to-air missiles effective from ~1km to ~8km depending on type
        # Aspect angle should be head-on or near head-on for best kill probability
        return (1500 <= distance <= 8000 and 
                abs(aspect_angle) < 30 and  # Tighter aspect for better missile performance
                locked > 0)

    def _in_optimal_range(self, distance: float, engagement_phase: str) -> bool:
        """Determine if in optimal range for current air combat phase"""
        if engagement_phase == "BVR":
            return 8000 <= distance <= 15000  # Long-range missile engagement
        elif engagement_phase == "INTERMEDIATE":
            return 4000 <= distance <= 8000   # Medium-range missile engagement
        elif engagement_phase == "WVR":
            return 1500 <= distance <= 4000   # Short-range missiles and guns
        elif engagement_phase == "KNIFE_FIGHT":
            return 500 <= distance <= 1500    # Guns only, high-G maneuvering
        else:
            return 2000 <= distance <= 6000   # General engagement range

    def _assess_pursuit_geometry(self, aspect_angle: float, closure_rate: float) -> str:
        """Assess pursuit curve geometry in air combat"""
        # In air combat, pursuit curves determine attack effectiveness
        if abs(aspect_angle) < 15 and closure_rate > 50:
            return "PURE_PURSUIT"      # Direct intercept, good for head-on shots
        elif 15 <= abs(aspect_angle) < 45 and closure_rate > 0:
            return "LEAD_PURSUIT"      # Leading target, good for gun attacks
        elif abs(aspect_angle) >= 45 or closure_rate < -20:
            return "LAG_PURSUIT"       # Falling behind, defensive position
        elif closure_rate < 0:
            return "DEFENSIVE_SPIRAL"  # Opponent gaining, need evasive action
        else:
            return "NEUTRAL"           # Parallel or maintaining distance

    def _assess_missile_zone(self, distance: float, locked: int, aspect_angle: float) -> str:
        """Assess missile employment zone for air-to-air engagement"""
        if locked <= 0:
            return "NO_LOCK"           # Cannot fire without radar lock
        elif distance > 15000:
            return "MAX_RANGE"         # Long-range missile max effective range
        elif distance > 8000 and abs(aspect_angle) < 20:
            return "BVR_OPTIMAL"       # Beyond visual range, optimal missile shot
        elif distance > 4000 and abs(aspect_angle) < 30:
            return "MEDIUM_RANGE"      # Medium-range missile engagement
        elif distance > 1500 and abs(aspect_angle) < 45:
            return "SHORT_RANGE"       # Short-range missile or guns
        elif distance > 800:
            return "GUNS_RANGE"        # Gun range, very close
        else:
            return "DANGER_CLOSE"      # Too close, risk of fratricide

    def _assess_defensive_urgency(self, threat_level: float, distance: float, closure_rate: float) -> str:
        """Assess defensive urgency level in air combat"""
        if threat_level > 0.8 or (distance < 1500 and closure_rate > 100):
            return "BREAK_BREAK"       # Immediate defensive maneuver required
        elif threat_level > 0.6 or (distance < 3000 and closure_rate > 50):
            return "DEFENSIVE"         # Need defensive maneuvering
        elif threat_level > 0.4 or (distance < 5000 and closure_rate > 0):
            return "CAUTION"           # Heightened awareness needed
        elif distance > 10000:
            return "SURVEILLANCE"      # Monitor but no immediate threat
        else:
            return "NEUTRAL"           # No immediate defensive action needed

    def _assess_energy_priority(self, energy_state: str, altitude: float, threat_level: float) -> str:
        """Assess energy management priority"""
        if energy_state == "LOW" and threat_level > 0.5:
            return "CRITICAL"
        elif energy_state == "LOW":
            return "HIGH"
        elif energy_state == "HIGH" and altitude > 8000:
            return "MAINTAIN"
        else:
            return "MODERATE"

    def request_shaping(self, features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Request tactical shaping from LLM with rate limiting"""
        
        # Rate limiting
        now = time.time()
        min_interval = 1.0 / max(self.max_rate_hz, 1e-3)
        if (now - self._last_time_s) < min_interval and self._last_features is not None:
            return self._last_response

        # Generate tactical prompt
        prompt = self._generate_tactical_prompt(features)
        
        try:
            resp = self.llm.invoke(prompt)
            text = resp.content if hasattr(resp, 'content') else str(resp)
            
            # Parse LLM response
            data = json.loads(text)
            shaping_delta = float(data.get("shaping_delta", 0.0))
            
            # Safety clamp
            shaping_delta = max(-0.5, min(0.5, shaping_delta))
            
            # Store for rate limiting
            self._last_time_s = now
            self._last_features = features
            self._last_response = (shaping_delta, data)
            
            # Log tactical feedback if verbose
            if self.verbose:
                self._log_tactical_feedback(features, data, shaping_delta)
            
            return self._last_response
            
        except Exception as e:
            if self.verbose:
                print(f"[TACTICAL ERROR] LLM parsing failed: {e}")
            
            # Fallback response
            self._last_time_s = now
            self._last_features = features
            fallback_response = (0.0, {
                "critique": f"llm_error: {str(e)[:100]}", 
                "shaping_delta": 0.0,
                "fallback": True
            })
            self._last_response = fallback_response
            return fallback_response

    def _generate_tactical_prompt(self, features: Dict[str, Any]) -> str:
        """Generate comprehensive tactical prompt for LLM"""
        
        prompt = f"""You are a TOP GUN instructor pilot providing tactical guidance for air-to-air combat.

MISSION: Guide an RL agent in fighter aircraft dogfighting using proven air combat doctrine and tactics.

=== TACTICAL PICTURE ===
Range: {features['distance']:.0f}m | Engagement: {features['engagement_band']}
BRA: {features['aspect_angle']:.1f}° aspect | Closure: {features['closure_rate']:.1f}m/s
Angels: {features['altitude']:.0f}m ({features['altitude_band']}) | Energy: {features['energy_state']}
Combat Phase: {features['engagement_phase']}

=== SENSORS & WEAPONS ===
Radar Lock: {'LOCKED' if features['locked'] > 0 else 'NO LOCK'} | Lock Time: {features['lock_duration']}s
Missile: {'READY' if features['missile1_state'] > 0 else 'EXPENDED'}
Bandit Status: {features['enemy_health']:.1%} integrity
Nose Position: {features['target_angle']:.3f} (0=nose-on, 1=tail-on)

=== AIRCRAFT STATUS ===
G-Loading: {features['g_force']:.1f}G | Turn Rate: {features['turn_rate']:.1f}°/s
Rate of Climb: {features['climb_rate']:.1f}m/s
Threat Level: {features['threat_level']:.1f} | Tactical Situation: {features['tactical_situation']}

=== BVR/ACM ANALYSIS ===
Pursuit Curve: {features['pursuit_geometry']}
Weapon Employment Zone: {features['missile_employment_zone']}
Defensive Posture: {features['defensive_urgency']}
Energy Management: {features['energy_management_priority']}
In WEZ (Weapon Engagement Zone): {'YES' if features['in_firing_envelope'] else 'NO'}
Optimal Engagement Range: {'YES' if features['optimal_range'] else 'NO'}

=== CONTROL INPUTS ===
Pitch: {features['action'][0]:.2f} | Roll: {features['action'][1]:.2f} | Rudder: {features['action'][2]:.2f} | Trigger: {features['action'][3]:.2f}
Control Smoothness: {features['action_smoothness']:.3f} (lower = smoother)

=== RULES OF ENGAGEMENT (ROE) ===
1. WEAPONS TIGHT - No shots without positive target identification and radar lock
2. ALTITUDE/ENERGY DISCIPLINE - Maintain energy advantage at all times
3. MUTUAL SUPPORT - Maintain proper formation and communication
4. DEFENSIVE AWARENESS - Maintain situational awareness of threats
5. CONTROLLED AGGRESSION - Smooth inputs, precise weapons employment

=== TACTICAL DOCTRINE ===
- BVR: First-look, first-shot, first-kill using long-range missiles
- MERGE: Proper merge geometry, avoid HCA (Head-on Attack)
- ACM: Maintain energy, use vertical plane, control range and aspect
- DEFENSIVE: Notch, chaff/flare, defensive BFM (Basic Fighter Maneuvers)
- OFFENSIVE: Lead pursuit, gun tracking, proper WEZ management

=== OUTPUT REQUIREMENT ===
Provide a JSON response with tactical assessment and reward shaping:

{{
  "shaping_delta": <float in [-0.5, 0.5]>,
  "critique": "<tactical assessment in <150 chars>",
  "tactical_assessment": {{
    "situation": "<tactical situation summary>",
    "priority": "<immediate tactical priority>",
    "risk_level": "<assessment of current risk>",
    "opportunity": "<current tactical opportunity if any>"
  }},
  "recommendations": {{
    "immediate": "<immediate action recommendation>",
    "tactical": "<broader tactical guidance>",
    "energy": "<energy management advice>",
    "weapons": "<weapons employment guidance>"
  }},
  "action_space_feedback": {{
    "effectiveness": "<assessment of last action effectiveness>",
    "improvements": ["<specific improvement suggestions>"]
  }}
}}

=== SHAPING GUIDANCE ===
- Positive (+0.1 to +0.5): Good tactical decisions, proper energy management, smart positioning
- Negative (-0.1 to -0.5): Tactical errors, energy waste, poor positioning, rules violations
- Zero (0.0): Neutral situations or when unsure

Be precise, tactical, and educational. This agent needs to learn fighter pilot fundamentals."""

        return prompt

    def _log_tactical_feedback(self, features: Dict[str, Any], llm_data: Dict[str, Any], shaping_delta: float):
        """Log tactical feedback for debugging and analysis"""
        
        situation = features.get('tactical_situation', 'UNKNOWN')
        engagement_band = features.get('engagement_band', 'UNKNOWN')
        critique = llm_data.get('critique', 'No critique')
        
        # Color coding for shaping delta
        if shaping_delta > 0.1:
            delta_str = f"[+] +{shaping_delta:.2f}"
        elif shaping_delta < -0.1:
            delta_str = f"[-] {shaping_delta:.2f}"
        else:
            delta_str = f"[=] {shaping_delta:.2f}"
        
        print(f"[TACTICAL] {situation} | {engagement_band} | {delta_str} | {critique}")
        
        # Log recommendations if available
        recommendations = llm_data.get('recommendations', {})
        if recommendations:
            immediate = recommendations.get('immediate', '')
            if immediate:
                print(f"[ADVICE] {immediate}")

    def get_tactical_summary(self) -> Dict[str, Any]:
        """Get summary of tactical guidance provided"""
        return {
            "total_guidance_calls": len(self.engagement_history),
            "last_features": self._last_features,
            "last_response": self._last_response[1],
            "rate_limit_active": (time.time() - self._last_time_s) < (1.0 / self.max_rate_hz)
        }


# Backwards compatibility function for existing code
def render_harfang_prompt(features: dict) -> str:
    """Backwards compatibility function - redirects to tactical assistant"""
    assistant = HarfangTacticalAssistant(llm=None, verbose=False)
    return assistant._generate_tactical_prompt(features)
