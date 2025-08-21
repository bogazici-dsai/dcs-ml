# Comprehensive Tactical Reward System for Harfang RL-LLM
import numpy as np
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RewardWeights:
    """Configurable reward weights for different tactical aspects"""
    # Core combat rewards
    survival: float = 1.0
    target_destruction: float = 100.0
    missile_hit: float = 50.0
    
    # Tactical positioning rewards
    lock_acquisition: float = 10.0
    lock_maintenance: float = 5.0
    optimal_range_bonus: float = 3.0
    energy_management: float = 2.0
    
    # Advanced tactical rewards
    bvr_positioning: float = 4.0
    wvr_maneuvering: float = 3.0
    defensive_effectiveness: float = 6.0
    offensive_efficiency: float = 4.0
    
    # Penalty weights
    ammunition_waste: float = -5.0
    poor_positioning: float = -2.0
    energy_waste: float = -1.0
    safety_violations: float = -10.0


class TacticalRewardSystem:
    """
    Comprehensive reward system combining base environment rewards with LLM tactical guidance
    and advanced tactical analysis for air combat training.
    """
    
    def __init__(self, llm_assistant, weights: RewardWeights = None):
        """
        Initialize tactical reward system
        
        Args:
            llm_assistant: HarfangTacticalAssistant for LLM guidance
            weights: Custom reward weights (uses defaults if None)
        """
        self.llm_assistant = llm_assistant
        self.weights = weights or RewardWeights()
        
        # Tactical knowledge for reward calculation
        self.optimal_engagement_ranges = {
            'BVR': (8000, 15000),      # Beyond Visual Range
            'INTERMEDIATE': (4000, 8000),  # Medium range
            'WVR': (1500, 4000),       # Within Visual Range
            'KNIFE_FIGHT': (500, 1500) # Close combat
        }
        
        # Performance tracking
        self.episode_metrics = {
            'shots_fired': 0,
            'shots_hit': 0,
            'locks_acquired': 0,
            'lock_time_total': 0,
            'energy_violations': 0,
            'tactical_score': 0.0
        }
        
        # Historical performance for adaptive rewards
        self.performance_history = []
        
        print("[REWARD SYSTEM] Tactical reward system initialized")
        print(f"[REWARD SYSTEM] Weights: Survival={self.weights.survival}, Victory={self.weights.target_destruction}")
    
    def calculate_comprehensive_reward(self, state: np.ndarray, action: np.ndarray,
                                     next_state: np.ndarray, info: Dict[str, Any],
                                     llm_feedback: Dict[str, Any]) -> float:
        """
        Calculate comprehensive reward combining multiple tactical factors
        
        Args:
            state: Previous state
            action: Action taken
            next_state: Resulting state
            info: Environment info dictionary
            llm_feedback: LLM tactical feedback
        
        Returns:
            Total reward value
        """
        
        # Extract tactical features
        features = self._extract_tactical_features(state, action, next_state, info)
        
        # 1. Base environment reward
        base_reward = info.get('base_reward', 0.0)
        
        # 2. LLM tactical shaping
        llm_shaping = float(llm_feedback.get('shaping_delta', 0.0))
        
        # 3. Tactical positioning rewards
        positioning_reward = self._calculate_positioning_reward(features)
        
        # 4. Combat effectiveness rewards
        combat_reward = self._calculate_combat_reward(features, info)
        
        # 5. Energy management rewards
        energy_reward = self._calculate_energy_reward(features)
        
        # 6. Safety and doctrine compliance
        safety_reward = self._calculate_safety_reward(features, action)
        
        # 7. Advanced tactical bonuses
        advanced_reward = self._calculate_advanced_tactical_reward(features, llm_feedback)
        
        # Combine all rewards
        total_reward = (
            base_reward +
            llm_shaping +
            positioning_reward +
            combat_reward +
            energy_reward +
            safety_reward +
            advanced_reward
        )
        
        # Update episode metrics
        self._update_episode_metrics(features, info, llm_feedback)
        
        # Log detailed reward breakdown if significant
        if abs(total_reward) > 1.0 or self.llm_assistant.verbose:
            self._log_reward_breakdown(
                base_reward, llm_shaping, positioning_reward, combat_reward,
                energy_reward, safety_reward, advanced_reward, total_reward
            )
        
        return total_reward
    
    def _extract_tactical_features(self, state: np.ndarray, action: np.ndarray,
                                  next_state: np.ndarray, info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract tactical features for reward calculation"""
        
        # Distance and geometry
        distance = info.get('distance', 0)
        locked = info.get('locked', False)
        threat_level = info.get('threat_level', 0)
        engagement_phase = info.get('engagement_phase', 'UNKNOWN')
        
        # State-based features
        prev_distance = math.sqrt(state[0]**2 + state[1]**2 + state[2]**2) if len(state) > 2 else distance
        curr_distance = math.sqrt(next_state[0]**2 + next_state[1]**2 + next_state[2]**2) if len(next_state) > 2 else distance
        
        closure_rate = prev_distance - curr_distance  # Positive = closing
        
        # Tactical assessments
        in_optimal_range = self._is_in_optimal_range(distance, engagement_phase)
        energy_state = info.get('energy_state', 'MEDIUM')
        
        return {
            'distance': distance,
            'prev_distance': prev_distance,
            'closure_rate': closure_rate,
            'locked': locked,
            'threat_level': threat_level,
            'engagement_phase': engagement_phase,
            'in_optimal_range': in_optimal_range,
            'energy_state': energy_state,
            'action_magnitude': np.linalg.norm(action[:3]),  # Control input magnitude
            'fire_action': action[3] if len(action) > 3 else 0.0
        }
    
    def _calculate_positioning_reward(self, features: Dict[str, Any]) -> float:
        """Calculate reward for tactical positioning"""
        reward = 0.0
        
        distance = features['distance']
        engagement_phase = features['engagement_phase']
        closure_rate = features['closure_rate']
        
        # Optimal range bonus
        if features['in_optimal_range']:
            reward += self.weights.optimal_range_bonus
        
        # Range management
        if engagement_phase == 'BVR':
            # BVR: Maintain standoff distance
            if 10000 <= distance <= 20000:
                reward += self.weights.bvr_positioning
            elif distance < 5000:
                reward -= self.weights.poor_positioning  # Too close for BVR
        
        elif engagement_phase == 'WVR':
            # WVR: Aggressive positioning for guns/short-range missiles
            if 2000 <= distance <= 5000:
                reward += self.weights.wvr_maneuvering
            elif distance > 8000:
                reward -= self.weights.poor_positioning  # Too far for WVR
        
        # Closure rate management
        if engagement_phase == 'BVR' and closure_rate > 200:
            reward -= 1.0  # Closing too fast in BVR
        elif engagement_phase == 'WVR' and closure_rate < -100:
            reward -= 1.0  # Opening range in WVR
        
        return reward
    
    def _calculate_combat_reward(self, features: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Calculate reward for combat effectiveness"""
        reward = 0.0
        
        # Lock acquisition and maintenance
        if features['locked']:
            reward += self.weights.lock_maintenance
            
            # Bonus for maintaining lock
            lock_duration = info.get('lock_duration', 0)
            if lock_duration > 5:
                reward += self.weights.lock_maintenance * 0.5
        
        # Missile firing assessment
        if features['fire_action'] > 0.5:
            if features['locked'] and features['in_optimal_range']:
                reward += self.weights.offensive_efficiency
                self.episode_metrics['shots_fired'] += 1
            else:
                reward += self.weights.ammunition_waste  # Wasted shot
        
        # Target destruction
        enemy_health = info.get('enemy_health', 1.0)
        if enemy_health <= 0:
            reward += self.weights.target_destruction
            self.episode_metrics['shots_hit'] += 1
        
        # Defensive effectiveness
        threat_level = features['threat_level']
        if threat_level > 0.6:
            # Under threat - reward defensive actions
            if features['action_magnitude'] > 0.3:  # Active maneuvering
                reward += self.weights.defensive_effectiveness
        
        return reward
    
    def _calculate_energy_reward(self, features: Dict[str, Any]) -> float:
        """Calculate reward for energy management"""
        reward = 0.0
        
        energy_state = features['energy_state']
        engagement_phase = features['engagement_phase']
        threat_level = features['threat_level']
        
        # Energy state appropriateness
        if engagement_phase == 'BVR':
            if energy_state == 'HIGH':
                reward += self.weights.energy_management  # Good energy for BVR
            elif energy_state == 'LOW':
                reward -= self.weights.energy_management  # Poor energy for BVR
        
        elif engagement_phase == 'WVR':
            if energy_state in ['HIGH', 'MEDIUM']:
                reward += self.weights.energy_management * 0.5  # Good energy for WVR
        
        # Energy conservation under low threat
        if threat_level < 0.3 and energy_state == 'HIGH':
            reward += self.weights.energy_management * 0.3  # Conserving energy
        
        return reward
    
    def _calculate_safety_reward(self, features: Dict[str, Any], action: np.ndarray) -> float:
        """Calculate reward for safety and doctrine compliance"""
        reward = 0.0
        
        # Control input smoothness (avoid excessive G-forces)
        action_magnitude = features['action_magnitude']
        if action_magnitude > 0.8:
            reward += self.weights.safety_violations * 0.5  # Excessive control inputs
        
        # Altitude safety (basic check)
        # Note: In real implementation, would check actual altitude from state
        
        # Engagement doctrine compliance
        distance = features['distance']
        engagement_phase = features['engagement_phase']
        
        # Don't fire missiles at extremely close range (fratricide risk)
        if features['fire_action'] > 0.5 and distance < 1000:
            reward += self.weights.safety_violations  # Dangerous shot
        
        # Don't engage in knife fight without proper energy
        if engagement_phase == 'KNIFE_FIGHT' and features['energy_state'] == 'LOW':
            reward += self.weights.safety_violations * 0.3  # Risky engagement
        
        return reward
    
    def _calculate_advanced_tactical_reward(self, features: Dict[str, Any], 
                                          llm_feedback: Dict[str, Any]) -> float:
        """Calculate advanced tactical rewards based on LLM assessment"""
        reward = 0.0
        
        # LLM tactical assessment bonuses
        tactical_assessment = llm_feedback.get('tactical_assessment', {})
        
        if isinstance(tactical_assessment, dict):
            situation = tactical_assessment.get('situation', '')
            priority = tactical_assessment.get('priority', '')
            
            # Reward for achieving tactical priorities
            if 'advantage' in situation.lower():
                reward += 2.0
            elif 'optimal' in situation.lower():
                reward += 1.0
            
            # Reward for following LLM recommendations
            recommendations = llm_feedback.get('recommendations', {})
            if isinstance(recommendations, dict):
                immediate = recommendations.get('immediate', '').lower()
                
                # Check if action aligns with LLM recommendation
                if self._action_aligns_with_recommendation(features, immediate):
                    reward += 1.5  # Bonus for following good advice
        
        return reward
    
    def _action_aligns_with_recommendation(self, features: Dict[str, Any], 
                                         recommendation: str) -> bool:
        """Check if current action aligns with LLM recommendation"""
        
        # Simple alignment check (can be enhanced)
        action_magnitude = features['action_magnitude']
        fire_action = features['fire_action']
        
        if 'fire' in recommendation and fire_action > 0.5:
            return True
        elif 'maneuver' in recommendation and action_magnitude > 0.3:
            return True
        elif 'maintain' in recommendation and action_magnitude < 0.2:
            return True
        
        return False
    
    def _is_in_optimal_range(self, distance: float, engagement_phase: str) -> bool:
        """Check if in optimal engagement range for current phase"""
        
        if engagement_phase in self.optimal_engagement_ranges:
            min_range, max_range = self.optimal_engagement_ranges[engagement_phase]
            return min_range <= distance <= max_range
        
        # Default optimal range
        return 3000 <= distance <= 8000
    
    def _update_episode_metrics(self, features: Dict[str, Any], info: Dict[str, Any],
                               llm_feedback: Dict[str, Any]):
        """Update episode-level metrics for analysis"""
        
        if features['locked'] and not hasattr(self, '_was_locked'):
            self.episode_metrics['locks_acquired'] += 1
        
        if features['locked']:
            self.episode_metrics['lock_time_total'] += 1
        
        if features['fire_action'] > 0.5:
            self.episode_metrics['shots_fired'] += 1
        
        # Track energy violations
        if features['energy_state'] == 'LOW' and features['engagement_phase'] == 'BVR':
            self.episode_metrics['energy_violations'] += 1
        
        # Update tactical score based on LLM feedback
        llm_shaping = float(llm_feedback.get('shaping_delta', 0.0))
        self.episode_metrics['tactical_score'] += llm_shaping
        
        # Store previous state for next comparison
        self._was_locked = features['locked']
    
    def _log_reward_breakdown(self, base_reward: float, llm_shaping: float,
                            positioning_reward: float, combat_reward: float,
                            energy_reward: float, safety_reward: float,
                            advanced_reward: float, total_reward: float):
        """Log detailed reward breakdown for analysis"""
        
        if abs(total_reward) > 5.0:  # Only log significant rewards
            print(f"[REWARD] Total: {total_reward:.2f} = "
                  f"Base:{base_reward:.1f} + LLM:{llm_shaping:.1f} + "
                  f"Pos:{positioning_reward:.1f} + Combat:{combat_reward:.1f} + "
                  f"Energy:{energy_reward:.1f} + Safety:{safety_reward:.1f} + "
                  f"Advanced:{advanced_reward:.1f}")
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get episode performance summary"""
        
        # Calculate derived metrics
        shot_accuracy = (self.episode_metrics['shots_hit'] / 
                        max(self.episode_metrics['shots_fired'], 1))
        
        lock_efficiency = (self.episode_metrics['locks_acquired'] / 
                          max(self.episode_metrics['lock_time_total'], 1))
        
        summary = {
            **self.episode_metrics,
            'shot_accuracy': shot_accuracy,
            'lock_efficiency': lock_efficiency,
            'tactical_effectiveness': self.episode_metrics['tactical_score'] / max(self.episode_metrics.get('steps', 1), 1)
        }
        
        return summary
    
    def reset_episode_metrics(self):
        """Reset metrics for new episode"""
        self.episode_metrics = {
            'shots_fired': 0,
            'shots_hit': 0,
            'locks_acquired': 0,
            'lock_time_total': 0,
            'energy_violations': 0,
            'tactical_score': 0.0,
            'steps': 0
        }
        
        # Reset tracking variables
        self._was_locked = False
    
    def adapt_weights_based_on_performance(self, episode_results: List[Dict[str, Any]]):
        """Adapt reward weights based on recent performance"""
        
        if len(episode_results) < 10:
            return  # Need sufficient data
        
        recent_results = episode_results[-10:]  # Last 10 episodes
        
        # Analyze performance trends
        avg_shot_accuracy = np.mean([r.get('shot_accuracy', 0) for r in recent_results])
        avg_tactical_score = np.mean([r.get('tactical_effectiveness', 0) for r in recent_results])
        
        # Adaptive weight adjustments
        if avg_shot_accuracy < 0.3:  # Poor shooting
            self.weights.lock_acquisition *= 1.1  # Increase lock rewards
            self.weights.ammunition_waste *= 1.2  # Increase waste penalty
            print("[REWARD ADAPT] Increased lock/accuracy focus")
        
        if avg_tactical_score < 0:  # Poor tactical decisions
            # Increase LLM guidance influence by reducing competing rewards
            self.weights.energy_management *= 0.9
            print("[REWARD ADAPT] Increased LLM guidance influence")
        
        print(f"[REWARD ADAPT] Adapted based on accuracy={avg_shot_accuracy:.2f}, tactical={avg_tactical_score:.2f}")


class CurriculumRewardSystem(TacticalRewardSystem):
    """
    Extended reward system with curriculum learning support
    """
    
    def __init__(self, llm_assistant, weights: RewardWeights = None):
        super().__init__(llm_assistant, weights)
        
        # Curriculum stages
        self.curriculum_stage = 0
        self.curriculum_stages = [
            {
                'name': 'basic_flight',
                'focus': 'flight_stability',
                'weight_multipliers': {'energy_management': 2.0, 'safety_violations': 2.0}
            },
            {
                'name': 'target_acquisition',
                'focus': 'lock_acquisition',
                'weight_multipliers': {'lock_acquisition': 2.0, 'lock_maintenance': 1.5}
            },
            {
                'name': 'bvr_engagement',
                'focus': 'long_range_combat',
                'weight_multipliers': {'bvr_positioning': 2.0, 'missile_hit': 1.5}
            },
            {
                'name': 'wvr_combat',
                'focus': 'dogfighting',
                'weight_multipliers': {'wvr_maneuvering': 2.0, 'defensive_effectiveness': 1.5}
            },
            {
                'name': 'advanced_tactics',
                'focus': 'complete_tactical_proficiency',
                'weight_multipliers': {}  # No modifications - balanced approach
            }
        ]
        
        print(f"[CURRICULUM REWARD] Initialized with {len(self.curriculum_stages)} stages")
    
    def advance_curriculum_stage(self, performance_metrics: Dict[str, float]):
        """Advance curriculum stage based on performance"""
        
        current_stage = self.curriculum_stages[self.curriculum_stage]
        
        # Stage-specific advancement criteria
        should_advance = False
        
        if current_stage['name'] == 'basic_flight':
            should_advance = performance_metrics.get('energy_violations', 10) < 2
        elif current_stage['name'] == 'target_acquisition':
            should_advance = performance_metrics.get('lock_efficiency', 0) > 0.3
        elif current_stage['name'] == 'bvr_engagement':
            should_advance = performance_metrics.get('shot_accuracy', 0) > 0.4
        elif current_stage['name'] == 'wvr_combat':
            should_advance = performance_metrics.get('tactical_effectiveness', 0) > 0.5
        
        if should_advance and self.curriculum_stage < len(self.curriculum_stages) - 1:
            self.curriculum_stage += 1
            new_stage = self.curriculum_stages[self.curriculum_stage]
            
            # Apply new weight multipliers
            self._apply_curriculum_weights(new_stage['weight_multipliers'])
            
            print(f"[CURRICULUM] Advanced to stage {self.curriculum_stage}: {new_stage['name']}")
            print(f"[CURRICULUM] Focus: {new_stage['focus']}")
    
    def _apply_curriculum_weights(self, multipliers: Dict[str, float]):
        """Apply curriculum-specific weight multipliers"""
        
        # Reset to base weights
        self.weights = RewardWeights()
        
        # Apply multipliers
        for weight_name, multiplier in multipliers.items():
            if hasattr(self.weights, weight_name):
                current_value = getattr(self.weights, weight_name)
                setattr(self.weights, weight_name, current_value * multiplier)
        
        print(f"[CURRICULUM] Applied weight multipliers: {multipliers}")


def create_reward_system(llm_assistant, use_curriculum: bool = False,
                        custom_weights: Dict[str, float] = None) -> TacticalRewardSystem:
    """
    Factory function to create appropriate reward system
    
    Args:
        llm_assistant: Tactical assistant for LLM guidance
        use_curriculum: Whether to use curriculum learning
        custom_weights: Custom reward weights
    
    Returns:
        Configured reward system
    """
    
    # Create custom weights if provided
    weights = RewardWeights()
    if custom_weights:
        for key, value in custom_weights.items():
            if hasattr(weights, key):
                setattr(weights, key, value)
    
    # Create appropriate reward system
    if use_curriculum:
        reward_system = CurriculumRewardSystem(llm_assistant, weights)
        print("[REWARD FACTORY] Created curriculum reward system")
    else:
        reward_system = TacticalRewardSystem(llm_assistant, weights)
        print("[REWARD FACTORY] Created standard reward system")
    
    return reward_system


if __name__ == "__main__":
    print("Tactical Reward System for Enhanced Harfang RL-LLM")
    print("Provides comprehensive reward calculation with LLM integration")
    print("Usage: Import and use with enhanced training system")
