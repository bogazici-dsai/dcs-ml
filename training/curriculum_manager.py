# Curriculum Learning Manager for Progressive Combat Training
import numpy as np
import json
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class CurriculumStage(Enum):
    """Curriculum learning stages for combat training"""
    BASIC_FLIGHT = "basic_flight"
    TARGET_ACQUISITION = "target_acquisition"
    BVR_ENGAGEMENT = "bvr_engagement"
    WVR_DOGFIGHT = "wvr_dogfight"
    DEFENSIVE_TACTICS = "defensive_tactics"
    ADVANCED_COMBAT = "advanced_combat"
    ACE_LEVEL = "ace_level"


@dataclass
class CurriculumStageConfig:
    """Configuration for a curriculum stage"""
    name: str
    duration_timesteps: int
    enemy_skill_level: float  # 0.0 to 1.0
    scenario_complexity: str  # 'simple', 'medium', 'complex', 'expert'
    success_threshold: float  # Required success rate to advance
    focus_areas: List[str]    # Areas of focus for this stage
    reward_multipliers: Dict[str, float]  # Reward weight adjustments
    environment_config: Dict[str, Any]    # Environment modifications
    description: str


class CombatCurriculum:
    """
    Progressive curriculum for training combat pilots from basic flight to ace-level combat.
    
    The curriculum follows real pilot training progression:
    1. Basic Flight - Learn aircraft control and stability
    2. Target Acquisition - Learn radar and targeting systems
    3. BVR Engagement - Beyond Visual Range missile combat
    4. WVR Dogfight - Within Visual Range air combat maneuvering
    5. Defensive Tactics - Evasion and defensive maneuvers
    6. Advanced Combat - Complex multi-threat scenarios
    7. Ace Level - Expert-level combat against skilled opponents
    """
    
    def __init__(self, total_training_timesteps: int = 5000000):
        """
        Initialize curriculum manager
        
        Args:
            total_training_timesteps: Total training time to distribute across stages
        """
        self.total_timesteps = total_training_timesteps
        self.current_stage = 0
        self.stage_start_time = 0
        self.stage_performance_history = []
        
        # Define curriculum stages based on real pilot training
        self.stages = self._create_curriculum_stages(total_training_timesteps)
        
        # Performance tracking
        self.advancement_history = []
        self.stage_metrics = {}
        
        print(f"[CURRICULUM] Initialized with {len(self.stages)} stages")
        print(f"[CURRICULUM] Total training: {total_training_timesteps:,} timesteps")
        self._print_curriculum_overview()
    
    def _create_curriculum_stages(self, total_timesteps: int) -> List[CurriculumStageConfig]:
        """Create progressive curriculum stages"""
        
        # Distribute timesteps across stages (progressive increase)
        stage_durations = [
            int(total_timesteps * 0.10),  # Basic flight: 10%
            int(total_timesteps * 0.15),  # Target acquisition: 15%
            int(total_timesteps * 0.20),  # BVR engagement: 20%
            int(total_timesteps * 0.20),  # WVR dogfight: 20%
            int(total_timesteps * 0.15),  # Defensive tactics: 15%
            int(total_timesteps * 0.15),  # Advanced combat: 15%
            int(total_timesteps * 0.05)   # Ace level: 5%
        ]
        
        stages = [
            CurriculumStageConfig(
                name="Basic Flight Control",
                duration_timesteps=stage_durations[0],
                enemy_skill_level=0.0,  # No enemy initially
                scenario_complexity="simple",
                success_threshold=0.8,  # 80% flight stability
                focus_areas=["flight_stability", "basic_control", "energy_management"],
                reward_multipliers={
                    "energy_management": 3.0,
                    "safety_violations": 3.0,
                    "control_smoothness": 2.0
                },
                environment_config={
                    "enemy_present": False,
                    "weather": "clear",
                    "altitude_range": (5000, 8000),
                    "initial_distance": None
                },
                description="Learn basic aircraft control, energy management, and flight stability"
            ),
            
            CurriculumStageConfig(
                name="Target Acquisition & Tracking",
                duration_timesteps=stage_durations[1],
                enemy_skill_level=0.1,  # Very passive enemy
                scenario_complexity="simple",
                success_threshold=0.6,  # 60% lock acquisition rate
                focus_areas=["radar_operation", "target_tracking", "lock_maintenance"],
                reward_multipliers={
                    "lock_acquisition": 3.0,
                    "lock_maintenance": 2.0,
                    "target_tracking": 2.0
                },
                environment_config={
                    "enemy_present": True,
                    "enemy_evasion": False,
                    "initial_distance": (8000, 12000),
                    "enemy_predictable": True
                },
                description="Learn radar operation, target acquisition, and lock maintenance"
            ),
            
            CurriculumStageConfig(
                name="BVR Missile Engagement",
                duration_timesteps=stage_durations[2],
                enemy_skill_level=0.3,  # Basic defensive maneuvers
                scenario_complexity="medium",
                success_threshold=0.4,  # 40% BVR success rate
                focus_areas=["bvr_tactics", "missile_employment", "range_management"],
                reward_multipliers={
                    "bvr_positioning": 2.5,
                    "missile_hit": 2.0,
                    "optimal_range_bonus": 2.0
                },
                environment_config={
                    "initial_distance": (12000, 20000),
                    "enemy_evasion": True,
                    "countermeasures": False,
                    "missile_types": ["long_range"]
                },
                description="Learn Beyond Visual Range missile engagement tactics"
            ),
            
            CurriculumStageConfig(
                name="WVR Dogfighting",
                duration_timesteps=stage_durations[3],
                enemy_skill_level=0.5,  # Competent dogfighter
                scenario_complexity="medium",
                success_threshold=0.35, # 35% WVR success rate
                focus_areas=["dogfighting", "energy_tactics", "guns_tracking"],
                reward_multipliers={
                    "wvr_maneuvering": 2.5,
                    "energy_management": 2.0,
                    "guns_effectiveness": 2.0
                },
                environment_config={
                    "initial_distance": (3000, 6000),
                    "merge_geometry": True,
                    "energy_fights": True,
                    "missile_types": ["short_range", "guns"]
                },
                description="Learn Within Visual Range air combat maneuvering and dogfighting"
            ),
            
            CurriculumStageConfig(
                name="Defensive Tactics",
                duration_timesteps=stage_durations[4],
                enemy_skill_level=0.6,  # Aggressive attacker
                scenario_complexity="complex",
                success_threshold=0.5,  # 50% survival rate when defensive
                focus_areas=["defensive_bfm", "threat_evasion", "countermeasures"],
                reward_multipliers={
                    "defensive_effectiveness": 3.0,
                    "survival": 2.0,
                    "threat_evasion": 2.0
                },
                environment_config={
                    "initial_advantage": "enemy",
                    "threat_level": "high",
                    "countermeasures": True,
                    "multiple_threats": False
                },
                description="Learn defensive Basic Fighter Maneuvers and threat evasion"
            ),
            
            CurriculumStageConfig(
                name="Advanced Multi-Threat Combat",
                duration_timesteps=stage_durations[5],
                enemy_skill_level=0.8,  # Skilled opponents
                scenario_complexity="complex",
                success_threshold=0.4,  # 40% success against multiple threats
                focus_areas=["multi_target", "situational_awareness", "prioritization"],
                reward_multipliers={
                    "target_prioritization": 2.0,
                    "situational_awareness": 2.0,
                    "multi_target_management": 2.0
                },
                environment_config={
                    "enemy_count": 2,
                    "threat_diversity": True,
                    "complex_geometry": True,
                    "time_pressure": True
                },
                description="Learn to engage multiple threats with tactical prioritization"
            ),
            
            CurriculumStageConfig(
                name="Ace-Level Combat",
                duration_timesteps=stage_durations[6],
                enemy_skill_level=1.0,  # Expert AI opponents
                scenario_complexity="expert",
                success_threshold=0.6,  # 60% success at ace level
                focus_areas=["expert_tactics", "adaptive_strategy", "mission_success"],
                reward_multipliers={},  # Balanced - no artificial boosts
                environment_config={
                    "enemy_skill": "ace",
                    "scenario_variety": "maximum",
                    "mission_objectives": True,
                    "realistic_constraints": True
                },
                description="Master-level combat against expert opponents with mission objectives"
            )
        ]
        
        return stages
    
    def _print_curriculum_overview(self):
        """Print curriculum overview"""
        print(f"\n{'='*80}")
        print("COMBAT PILOT CURRICULUM OVERVIEW")
        print(f"{'='*80}")
        
        for i, stage in enumerate(self.stages):
            duration_pct = (stage.duration_timesteps / self.total_timesteps) * 100
            print(f"{i+1}. {stage.name}")
            print(f"   Duration: {stage.duration_timesteps:,} timesteps ({duration_pct:.1f}%)")
            print(f"   Enemy Skill: {stage.enemy_skill_level:.1f}/1.0")
            print(f"   Success Threshold: {stage.success_threshold:.1%}")
            print(f"   Focus: {', '.join(stage.focus_areas)}")
            print(f"   Description: {stage.description}")
            print()
    
    def get_current_stage(self) -> CurriculumStageConfig:
        """Get current curriculum stage configuration"""
        return self.stages[self.current_stage]
    
    def should_advance_stage(self, recent_performance: Dict[str, float]) -> bool:
        """
        Check if agent should advance to next curriculum stage
        
        Args:
            recent_performance: Recent performance metrics
        
        Returns:
            True if should advance to next stage
        """
        if self.current_stage >= len(self.stages) - 1:
            return False  # Already at final stage
        
        current_stage = self.get_current_stage()
        
        # Check stage-specific advancement criteria
        success_rate = recent_performance.get('success_rate', 0.0)
        
        # Stage-specific additional criteria
        additional_criteria_met = True
        
        if current_stage.name == "Basic Flight Control":
            # Need stable flight
            energy_violations = recent_performance.get('energy_violations', 10)
            control_smoothness = recent_performance.get('control_smoothness', 0)
            additional_criteria_met = energy_violations < 2 and control_smoothness > 0.7
        
        elif current_stage.name == "Target Acquisition & Tracking":
            # Need consistent lock acquisition
            lock_rate = recent_performance.get('lock_acquisition_rate', 0)
            additional_criteria_met = lock_rate > 0.5
        
        elif current_stage.name == "BVR Missile Engagement":
            # Need reasonable BVR performance
            shot_accuracy = recent_performance.get('shot_accuracy', 0)
            additional_criteria_met = shot_accuracy > 0.25
        
        elif current_stage.name == "WVR Dogfighting":
            # Need WVR survival and offense
            wvr_survival = recent_performance.get('wvr_survival_rate', 0)
            additional_criteria_met = wvr_survival > 0.4
        
        # Check if advancement criteria met
        should_advance = (success_rate >= current_stage.success_threshold and 
                         additional_criteria_met)
        
        if should_advance:
            print(f"[CURRICULUM] Stage advancement criteria met!")
            print(f"   Success rate: {success_rate:.1%} >= {current_stage.success_threshold:.1%}")
            print(f"   Additional criteria: {additional_criteria_met}")
        
        return should_advance
    
    def advance_to_next_stage(self) -> bool:
        """Advance to next curriculum stage"""
        
        if self.current_stage >= len(self.stages) - 1:
            print("[CURRICULUM] Already at final stage")
            return False
        
        # Record advancement
        old_stage = self.stages[self.current_stage]
        self.current_stage += 1
        new_stage = self.stages[self.current_stage]
        
        advancement_record = {
            'timestamp': time.time(),
            'from_stage': old_stage.name,
            'to_stage': new_stage.name,
            'stage_number': self.current_stage,
            'training_timesteps_completed': self.stage_start_time
        }
        
        self.advancement_history.append(advancement_record)
        self.stage_start_time = 0  # Reset stage timer
        
        print(f"\n{'='*60}")
        print(f"CURRICULUM ADVANCEMENT")
        print(f"{'='*60}")
        print(f"Advanced from: {old_stage.name}")
        print(f"Advanced to: {new_stage.name}")
        print(f"New focus: {', '.join(new_stage.focus_areas)}")
        print(f"Enemy skill: {new_stage.enemy_skill_level:.1f}/1.0")
        print(f"Success threshold: {new_stage.success_threshold:.1%}")
        print(f"Description: {new_stage.description}")
        print(f"{'='*60}")
        
        return True
    
    def get_stage_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration for current stage"""
        current_stage = self.get_current_stage()
        
        base_config = {
            'enemy_skill_level': current_stage.enemy_skill_level,
            'scenario_complexity': current_stage.scenario_complexity,
            'focus_areas': current_stage.focus_areas,
            'stage_name': current_stage.name
        }
        
        # Add stage-specific environment modifications
        base_config.update(current_stage.environment_config)
        
        return base_config
    
    def get_stage_reward_multipliers(self) -> Dict[str, float]:
        """Get reward multipliers for current stage"""
        current_stage = self.get_current_stage()
        return current_stage.reward_multipliers.copy()
    
    def update_stage_progress(self, timesteps_completed: int):
        """Update progress within current stage"""
        self.stage_start_time = timesteps_completed
        
        current_stage = self.get_current_stage()
        progress = min(1.0, timesteps_completed / current_stage.duration_timesteps)
        
        if timesteps_completed % 50000 == 0:  # Log every 50k timesteps
            print(f"[CURRICULUM] Stage '{current_stage.name}' progress: {progress:.1%}")
    
    def is_stage_complete(self, timesteps_completed: int) -> bool:
        """Check if current stage duration is complete"""
        current_stage = self.get_current_stage()
        return timesteps_completed >= current_stage.duration_timesteps
    
    def get_curriculum_summary(self) -> Dict[str, Any]:
        """Get comprehensive curriculum summary"""
        
        current_stage = self.get_current_stage()
        
        summary = {
            'total_stages': len(self.stages),
            'current_stage_number': self.current_stage + 1,
            'current_stage_name': current_stage.name,
            'current_stage_progress': self.stage_start_time / current_stage.duration_timesteps,
            'stages_completed': self.current_stage,
            'advancement_history': self.advancement_history,
            'stage_details': {
                'name': current_stage.name,
                'enemy_skill': current_stage.enemy_skill_level,
                'complexity': current_stage.scenario_complexity,
                'success_threshold': current_stage.success_threshold,
                'focus_areas': current_stage.focus_areas,
                'description': current_stage.description
            }
        }
        
        return summary
    
    def export_curriculum_progress(self, filepath: str = "logs/curriculum_progress.json"):
        """Export curriculum progress for analysis"""
        
        progress_data = {
            'curriculum_summary': self.get_curriculum_summary(),
            'stage_performance_history': self.stage_performance_history,
            'advancement_history': self.advancement_history,
            'export_timestamp': time.time()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        print(f"[CURRICULUM] Progress exported to: {filepath}")


class CurriculumTrainingManager:
    """
    Manager that integrates curriculum learning with RL training
    """
    
    def __init__(self, curriculum: CombatCurriculum, rl_agent, environment, 
                 reward_system, performance_evaluator: Callable):
        """
        Initialize curriculum training manager
        
        Args:
            curriculum: Curriculum definition
            rl_agent: RL agent to train
            environment: Training environment
            reward_system: Reward calculation system
            performance_evaluator: Function to evaluate agent performance
        """
        self.curriculum = curriculum
        self.rl_agent = rl_agent
        self.environment = environment
        self.reward_system = reward_system
        self.performance_evaluator = performance_evaluator
        
        # Training state
        self.total_timesteps_trained = 0
        self.stage_timesteps_trained = 0
        self.evaluation_frequency = 25000  # Evaluate every 25k timesteps
        
        print("[CURRICULUM MANAGER] Initialized curriculum training")
    
    def train_with_curriculum(self, total_timesteps: int, 
                            evaluation_episodes: int = 20) -> Dict[str, Any]:
        """
        Train agent using curriculum learning
        
        Args:
            total_timesteps: Total training timesteps
            evaluation_episodes: Episodes for stage evaluation
        
        Returns:
            Training results summary
        """
        
        print(f"\n{'='*80}")
        print("CURRICULUM-BASED TRAINING STARTED")
        print(f"{'='*80}")
        
        training_results = {
            'stages_completed': [],
            'advancement_history': [],
            'final_performance': {},
            'training_time': 0
        }
        
        start_time = time.time()
        
        while self.total_timesteps_trained < total_timesteps:
            current_stage = self.curriculum.get_current_stage()
            
            print(f"\n[STAGE] Training '{current_stage.name}' "
                  f"(Stage {self.curriculum.current_stage + 1}/{len(self.curriculum.stages)})")
            
            # Configure environment for current stage
            stage_config = self.curriculum.get_stage_environment_config()
            self._configure_environment_for_stage(stage_config)
            
            # Configure reward system for current stage
            stage_multipliers = self.curriculum.get_stage_reward_multipliers()
            self._configure_rewards_for_stage(stage_multipliers)
            
            # Calculate timesteps for this stage iteration
            stage_remaining = current_stage.duration_timesteps - self.stage_timesteps_trained
            training_chunk = min(self.evaluation_frequency, stage_remaining, 
                               total_timesteps - self.total_timesteps_trained)
            
            # Train for this chunk
            print(f"[TRAINING] Training for {training_chunk:,} timesteps...")
            self._train_stage_chunk(training_chunk)
            
            # Update progress
            self.total_timesteps_trained += training_chunk
            self.stage_timesteps_trained += training_chunk
            self.curriculum.update_stage_progress(self.stage_timesteps_trained)
            
            # Evaluate performance
            if self.total_timesteps_trained % self.evaluation_frequency == 0:
                performance = self.performance_evaluator(evaluation_episodes)
                print(f"[EVALUATION] Performance: {performance.get('success_rate', 0):.1%} success")
                
                # Check for stage advancement
                if self.curriculum.should_advance_stage(performance):
                    if self.curriculum.advance_to_next_stage():
                        # Record stage completion
                        stage_result = {
                            'stage_name': current_stage.name,
                            'timesteps_trained': self.stage_timesteps_trained,
                            'final_performance': performance,
                            'advancement_timestamp': time.time()
                        }
                        training_results['stages_completed'].append(stage_result)
                        
                        # Reset stage timer
                        self.stage_timesteps_trained = 0
                
                # Store performance history
                self.curriculum.stage_performance_history.append({
                    'timesteps': self.total_timesteps_trained,
                    'stage': current_stage.name,
                    'performance': performance
                })
            
            # Check if stage duration completed (forced advancement)
            if self.curriculum.is_stage_complete(self.stage_timesteps_trained):
                print(f"[CURRICULUM] Stage duration complete, forcing advancement")
                if self.curriculum.advance_to_next_stage():
                    self.stage_timesteps_trained = 0
        
        training_time = time.time() - start_time
        training_results['training_time'] = training_time
        
        # Final evaluation
        final_performance = self.performance_evaluator(evaluation_episodes * 2)
        training_results['final_performance'] = final_performance
        
        print(f"\n{'='*80}")
        print("CURRICULUM TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Total training time: {training_time/3600:.1f} hours")
        print(f"Stages completed: {len(training_results['stages_completed'])}/{len(self.curriculum.stages)}")
        print(f"Final success rate: {final_performance.get('success_rate', 0):.1%}")
        
        return training_results
    
    def _configure_environment_for_stage(self, stage_config: Dict[str, Any]):
        """Configure environment for current curriculum stage"""
        
        # Apply stage-specific environment settings
        if hasattr(self.environment, 'set_curriculum_config'):
            self.environment.set_curriculum_config(stage_config)
        else:
            # For mock environment, just log the configuration
            print(f"[ENV CONFIG] Stage config: {stage_config}")
    
    def _configure_rewards_for_stage(self, multipliers: Dict[str, float]):
        """Configure reward system for current curriculum stage"""
        
        if hasattr(self.reward_system, 'apply_curriculum_multipliers'):
            self.reward_system.apply_curriculum_multipliers(multipliers)
        else:
            # For basic reward system, just log
            if multipliers:
                print(f"[REWARD CONFIG] Stage multipliers: {multipliers}")
    
    def _train_stage_chunk(self, timesteps: int):
        """Train for a chunk of timesteps within current stage"""
        
        # This would integrate with the actual RL training loop
        # For now, simulate training
        if hasattr(self.rl_agent, 'learn'):
            self.rl_agent.learn(total_timesteps=timesteps)
        else:
            print(f"[TRAINING] Simulated training for {timesteps:,} timesteps")


def create_combat_curriculum(total_timesteps: int = 5000000,
                           difficulty_progression: str = "standard") -> CombatCurriculum:
    """
    Factory function to create combat curriculum
    
    Args:
        total_timesteps: Total training timesteps
        difficulty_progression: 'gentle', 'standard', 'aggressive'
    
    Returns:
        Configured combat curriculum
    """
    
    curriculum = CombatCurriculum(total_timesteps)
    
    # Adjust difficulty progression if requested
    if difficulty_progression == "gentle":
        # Extend early stages, lower thresholds
        for stage in curriculum.stages[:4]:
            stage.success_threshold *= 0.8
            stage.duration_timesteps = int(stage.duration_timesteps * 1.2)
    
    elif difficulty_progression == "aggressive":
        # Shorter stages, higher thresholds
        for stage in curriculum.stages:
            stage.success_threshold *= 1.1
            stage.duration_timesteps = int(stage.duration_timesteps * 0.8)
    
    print(f"[CURRICULUM FACTORY] Created {difficulty_progression} difficulty curriculum")
    return curriculum


if __name__ == "__main__":
    # Example curriculum creation and overview
    curriculum = create_combat_curriculum(total_timesteps=2000000, difficulty_progression="standard")
    
    print("\nCurriculum created successfully!")
    print("Use with CurriculumTrainingManager for progressive training")
