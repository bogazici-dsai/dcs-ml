# Mission Planning System for Context-Aware Combat Training
import random
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class MissionType(Enum):
    """Types of air combat missions"""
    AIR_SUPERIORITY = "air_superiority"
    INTERCEPT = "intercept"
    ESCORT = "escort"
    CAS = "cas"  # Close Air Support
    SEAD = "sead"  # Suppression of Enemy Air Defenses
    CAP = "cap"  # Combat Air Patrol
    STRIKE = "strike"
    RECONNAISSANCE = "reconnaissance"


class ThreatLevel(Enum):
    """Mission threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MissionObjective:
    """Individual mission objective"""
    type: str
    description: str
    priority: str  # 'primary', 'secondary', 'tertiary'
    success_criteria: Dict[str, Any]
    time_limit: Optional[int] = None  # seconds
    location: Optional[Tuple[float, float]] = None  # (lat, lon)


@dataclass
class ThreatAssessment:
    """Mission threat assessment"""
    air_threats: List[Dict[str, Any]]
    ground_threats: List[Dict[str, Any]]
    electronic_threats: List[Dict[str, Any]]
    overall_threat_level: ThreatLevel
    threat_density: float  # 0.0 to 1.0


@dataclass
class Mission:
    """Complete mission definition"""
    mission_id: str
    mission_type: MissionType
    objectives: List[MissionObjective]
    threat_assessment: ThreatAssessment
    rules_of_engagement: Dict[str, Any]
    time_constraints: Dict[str, int]
    weather_conditions: Dict[str, Any]
    friendly_assets: List[Dict[str, Any]]
    success_criteria: Dict[str, float]
    briefing: str


class MissionPlanner:
    """
    Advanced mission planning system that generates diverse, realistic combat missions
    for context-aware RL training. Provides strategic context that influences
    tactical decision-making throughout the engagement.
    """
    
    def __init__(self, llm_assistant=None, verbose: bool = True):
        """
        Initialize mission planner
        
        Args:
            llm_assistant: Optional LLM for mission briefing generation
            verbose: Enable detailed logging
        """
        self.llm_assistant = llm_assistant
        self.verbose = verbose
        
        # Mission templates and parameters
        self.aircraft_types = {
            'friendly': ['F-16C', 'F-18C', 'F-35A', 'F-22A'],
            'enemy': ['Su-27', 'Su-30', 'Su-35', 'MiG-29', 'J-20']
        }
        
        self.weapon_systems = {
            'air_to_air': ['AIM-120C', 'AIM-9X', 'AIM-7M', 'R-77', 'R-73'],
            'air_to_ground': ['AGM-65', 'AGM-88', 'GBU-12', 'GBU-31'],
            'guns': ['M61A1', 'GSh-30-1']
        }
        
        self.weather_conditions = [
            {'condition': 'clear', 'visibility': 'unlimited', 'impact': 'none'},
            {'condition': 'cloudy', 'visibility': '10km', 'impact': 'radar_degraded'},
            {'condition': 'overcast', 'visibility': '5km', 'impact': 'visual_limited'},
            {'condition': 'rain', 'visibility': '3km', 'impact': 'sensors_degraded'},
            {'condition': 'night', 'visibility': '1km', 'impact': 'visual_combat_limited'}
        ]
        
        # Mission generation statistics
        self.missions_generated = 0
        self.mission_history = []
        
        print(f"[MISSION PLANNER] Initialized with {len(MissionType)} mission types")
        if llm_assistant:
            print(f"[MISSION PLANNER] LLM briefing generation enabled")
    
    def generate_mission(self, mission_type: MissionType = None, 
                        difficulty: str = "medium",
                        custom_params: Dict[str, Any] = None) -> Mission:
        """
        Generate a comprehensive mission with objectives, threats, and constraints
        
        Args:
            mission_type: Specific mission type (random if None)
            difficulty: Mission difficulty ('easy', 'medium', 'hard', 'expert')
            custom_params: Custom mission parameters
        
        Returns:
            Complete mission definition
        """
        
        # Select mission type
        if mission_type is None:
            mission_type = random.choice(list(MissionType))
        
        mission_id = f"{mission_type.value}_{int(time.time())}_{self.missions_generated}"
        
        print(f"[MISSION GEN] Generating {mission_type.value} mission (difficulty: {difficulty})")
        
        # Generate mission components
        objectives = self._generate_objectives(mission_type, difficulty)
        threat_assessment = self._generate_threat_assessment(mission_type, difficulty)
        roe = self._generate_rules_of_engagement(mission_type, threat_assessment)
        time_constraints = self._generate_time_constraints(mission_type, difficulty)
        weather = random.choice(self.weather_conditions)
        friendly_assets = self._generate_friendly_assets(mission_type)
        success_criteria = self._generate_success_criteria(objectives, difficulty)
        
        # Generate mission briefing
        briefing = self._generate_mission_briefing(
            mission_type, objectives, threat_assessment, roe, weather
        )
        
        # Create mission object
        mission = Mission(
            mission_id=mission_id,
            mission_type=mission_type,
            objectives=objectives,
            threat_assessment=threat_assessment,
            rules_of_engagement=roe,
            time_constraints=time_constraints,
            weather_conditions=weather,
            friendly_assets=friendly_assets,
            success_criteria=success_criteria,
            briefing=briefing
        )
        
        # Update statistics
        self.missions_generated += 1
        self.mission_history.append({
            'mission_id': mission_id,
            'type': mission_type.value,
            'difficulty': difficulty,
            'generation_time': time.time()
        })
        
        if self.verbose:
            print(f"[MISSION GEN] Generated mission: {mission_id}")
            print(f"[MISSION GEN] Objectives: {len(objectives)}")
            print(f"[MISSION GEN] Threat level: {threat_assessment.overall_threat_level.value}")
        
        return mission
    
    def _generate_objectives(self, mission_type: MissionType, difficulty: str) -> List[MissionObjective]:
        """Generate mission-specific objectives"""
        
        objectives = []
        
        if mission_type == MissionType.AIR_SUPERIORITY:
            objectives.append(MissionObjective(
                type="destroy_enemy_aircraft",
                description="Neutralize all enemy aircraft in the area of operations",
                priority="primary",
                success_criteria={"enemy_aircraft_destroyed": 1, "own_aircraft_survival": True}
            ))
            
            if difficulty in ['hard', 'expert']:
                objectives.append(MissionObjective(
                    type="area_denial",
                    description="Deny enemy access to specified airspace",
                    priority="secondary",
                    success_criteria={"area_controlled_time": 300}  # 5 minutes
                ))
        
        elif mission_type == MissionType.INTERCEPT:
            objectives.append(MissionObjective(
                type="intercept_target",
                description="Intercept and neutralize incoming enemy aircraft",
                priority="primary", 
                success_criteria={"intercept_success": True, "target_neutralized": True},
                time_limit=600  # 10 minutes
            ))
        
        elif mission_type == MissionType.ESCORT:
            objectives.append(MissionObjective(
                type="protect_asset",
                description="Escort friendly aircraft to target and back",
                priority="primary",
                success_criteria={"asset_survival": True, "mission_completion": True}
            ))
            
            objectives.append(MissionObjective(
                type="threat_neutralization",
                description="Neutralize threats to escorted asset",
                priority="secondary",
                success_criteria={"threats_neutralized": True}
            ))
        
        elif mission_type == MissionType.CAP:
            objectives.append(MissionObjective(
                type="area_patrol",
                description="Maintain combat air patrol over designated area",
                priority="primary",
                success_criteria={"patrol_time": 1800, "area_security": True}  # 30 minutes
            ))
        
        # Add difficulty-based secondary objectives
        if difficulty in ['hard', 'expert']:
            objectives.append(MissionObjective(
                type="minimize_casualties",
                description="Complete mission with minimal friendly losses",
                priority="secondary",
                success_criteria={"friendly_losses": 0}
            ))
        
        return objectives
    
    def _generate_threat_assessment(self, mission_type: MissionType, difficulty: str) -> ThreatAssessment:
        """Generate realistic threat assessment for mission"""
        
        # Base threat levels by difficulty
        threat_multipliers = {
            'easy': 0.3,
            'medium': 0.6,
            'hard': 0.8,
            'expert': 1.0
        }
        
        base_threat = threat_multipliers.get(difficulty, 0.6)
        
        # Generate air threats
        air_threats = []
        num_air_threats = random.randint(1, 3) if difficulty != 'easy' else 1
        
        for i in range(num_air_threats):
            threat = {
                'type': 'enemy_aircraft',
                'aircraft_type': random.choice(self.aircraft_types['enemy']),
                'skill_level': base_threat + random.uniform(-0.1, 0.1),
                'weapons': random.sample(self.weapon_systems['air_to_air'], 2),
                'threat_rating': base_threat
            }
            air_threats.append(threat)
        
        # Generate ground threats (for some mission types)
        ground_threats = []
        if mission_type in [MissionType.CAS, MissionType.SEAD, MissionType.STRIKE]:
            num_ground_threats = random.randint(0, 2)
            for i in range(num_ground_threats):
                threat = {
                    'type': 'sam_site',
                    'system_type': random.choice(['SA-10', 'SA-15', 'SA-20']),
                    'threat_rating': base_threat * 0.7,
                    'range': random.randint(30, 100)  # km
                }
                ground_threats.append(threat)
        
        # Determine overall threat level
        max_threat = max([t['threat_rating'] for t in air_threats + ground_threats])
        if max_threat >= 0.8:
            overall_threat = ThreatLevel.CRITICAL
        elif max_threat >= 0.6:
            overall_threat = ThreatLevel.HIGH
        elif max_threat >= 0.4:
            overall_threat = ThreatLevel.MEDIUM
        else:
            overall_threat = ThreatLevel.LOW
        
        return ThreatAssessment(
            air_threats=air_threats,
            ground_threats=ground_threats,
            electronic_threats=[],  # Can be expanded
            overall_threat_level=overall_threat,
            threat_density=base_threat
        )
    
    def _generate_rules_of_engagement(self, mission_type: MissionType, 
                                    threat_assessment: ThreatAssessment) -> Dict[str, Any]:
        """Generate appropriate rules of engagement"""
        
        base_roe = {
            'weapons_status': 'WEAPONS_TIGHT',  # Default: positive ID required
            'self_defense': True,
            'collateral_damage_restrictions': True,
            'altitude_restrictions': None,
            'geographic_restrictions': None
        }
        
        # Adjust ROE based on mission type and threat level
        if mission_type == MissionType.AIR_SUPERIORITY:
            if threat_assessment.overall_threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                base_roe['weapons_status'] = 'WEAPONS_FREE'  # Engage any hostile
        
        elif mission_type == MissionType.INTERCEPT:
            base_roe['weapons_status'] = 'WEAPONS_FREE'  # Time-critical
            base_roe['pursuit_authorized'] = True
        
        elif mission_type == MissionType.ESCORT:
            base_roe['weapons_status'] = 'WEAPONS_TIGHT'  # Protect asset priority
            base_roe['formation_restrictions'] = True
        
        return base_roe
    
    def _generate_time_constraints(self, mission_type: MissionType, difficulty: str) -> Dict[str, int]:
        """Generate mission time constraints"""
        
        base_times = {
            MissionType.AIR_SUPERIORITY: {'mission_duration': 1800, 'engagement_window': 600},
            MissionType.INTERCEPT: {'mission_duration': 900, 'intercept_window': 300},
            MissionType.ESCORT: {'mission_duration': 2400, 'escort_duration': 1800},
            MissionType.CAP: {'mission_duration': 3600, 'patrol_duration': 3000},
            MissionType.CAS: {'mission_duration': 1200, 'support_window': 900}
        }
        
        base_time = base_times.get(mission_type, {'mission_duration': 1800})
        
        # Adjust for difficulty
        if difficulty == 'easy':
            for key in base_time:
                base_time[key] = int(base_time[key] * 1.5)  # More time
        elif difficulty == 'expert':
            for key in base_time:
                base_time[key] = int(base_time[key] * 0.7)  # Less time
        
        return base_time
    
    def _generate_friendly_assets(self, mission_type: MissionType) -> List[Dict[str, Any]]:
        """Generate friendly assets for mission"""
        
        assets = []
        
        # Always include player aircraft
        assets.append({
            'type': 'player_aircraft',
            'aircraft_type': 'F-16C',
            'role': 'primary_fighter',
            'weapons': ['AIM-120C', 'AIM-9X'],
            'fuel_state': 'full'
        })
        
        # Add mission-specific assets
        if mission_type == MissionType.ESCORT:
            assets.append({
                'type': 'escorted_asset',
                'aircraft_type': 'KC-135',  # Tanker
                'role': 'mission_asset',
                'protection_priority': 'highest'
            })
        
        elif mission_type in [MissionType.CAP, MissionType.AIR_SUPERIORITY]:
            # Add wingman
            assets.append({
                'type': 'wingman',
                'aircraft_type': 'F-16C',
                'role': 'support_fighter',
                'weapons': ['AIM-120C', 'AIM-9X']
            })
        
        return assets
    
    def _generate_success_criteria(self, objectives: List[MissionObjective], 
                                 difficulty: str) -> Dict[str, float]:
        """Generate mission success criteria"""
        
        base_criteria = {
            'primary_objectives_completed': 1.0,  # All primary objectives
            'secondary_objectives_completed': 0.5,  # 50% of secondary
            'survival_rate': 1.0,  # Survive the mission
            'collateral_damage': 0.0,  # No collateral damage
            'ammunition_efficiency': 0.6  # 60% hit rate
        }
        
        # Adjust for difficulty
        if difficulty == 'easy':
            base_criteria['primary_objectives_completed'] = 0.8
            base_criteria['ammunition_efficiency'] = 0.4
        elif difficulty == 'expert':
            base_criteria['secondary_objectives_completed'] = 0.8
            base_criteria['ammunition_efficiency'] = 0.8
        
        return base_criteria
    
    def _generate_mission_briefing(self, mission_type: MissionType,
                                 objectives: List[MissionObjective],
                                 threat_assessment: ThreatAssessment,
                                 roe: Dict[str, Any],
                                 weather: Dict[str, Any]) -> str:
        """Generate comprehensive mission briefing"""
        
        # Basic briefing structure
        briefing_sections = []
        
        # Mission overview
        briefing_sections.append(f"MISSION TYPE: {mission_type.value.upper()}")
        briefing_sections.append(f"THREAT LEVEL: {threat_assessment.overall_threat_level.value.upper()}")
        briefing_sections.append(f"WEATHER: {weather['condition'].title()} (Visibility: {weather['visibility']})")
        
        # Objectives
        briefing_sections.append("\nOBJECTIVES:")
        for i, obj in enumerate(objectives, 1):
            briefing_sections.append(f"{i}. {obj.description} ({obj.priority.upper()})")
        
        # Threat assessment
        briefing_sections.append(f"\nTHREAT ASSESSMENT:")
        briefing_sections.append(f"- Air Threats: {len(threat_assessment.air_threats)} aircraft")
        for threat in threat_assessment.air_threats:
            briefing_sections.append(f"  * {threat['aircraft_type']} (Skill: {threat['skill_level']:.1f})")
        
        if threat_assessment.ground_threats:
            briefing_sections.append(f"- Ground Threats: {len(threat_assessment.ground_threats)} sites")
            for threat in threat_assessment.ground_threats:
                briefing_sections.append(f"  * {threat['system_type']} SAM ({threat['range']}km range)")
        
        # Rules of engagement
        briefing_sections.append(f"\nRULES OF ENGAGEMENT:")
        briefing_sections.append(f"- Weapons Status: {roe['weapons_status']}")
        briefing_sections.append(f"- Self Defense: {'Authorized' if roe['self_defense'] else 'Restricted'}")
        
        # Enhanced briefing with LLM
        basic_briefing = "\n".join(briefing_sections)
        
        if self.llm_assistant and hasattr(self.llm_assistant, 'llm'):
            try:
                enhanced_briefing = self._generate_llm_briefing(basic_briefing, mission_type)
                return enhanced_briefing
            except Exception as e:
                print(f"[BRIEFING] LLM briefing failed: {e}")
                return basic_briefing
        
        return basic_briefing
    
    def _generate_llm_briefing(self, basic_briefing: str, mission_type: MissionType) -> str:
        """Generate enhanced briefing using LLM"""
        
        briefing_prompt = f"""You are a military briefing officer providing a tactical mission briefing.

Basic Mission Information:
{basic_briefing}

Enhance this briefing with:
1. Tactical considerations and recommendations
2. Potential threats and countermeasures
3. Success factors and key risks
4. Recommended approach and tactics

Provide a professional, concise military briefing in the following format:

MISSION BRIEFING - {mission_type.value.upper()}

[Enhanced briefing content]

TACTICAL RECOMMENDATIONS:
- [Key tactical recommendations]

RISK ASSESSMENT:
- [Primary risks and mitigation strategies]

MISSION SUCCESS FACTORS:
- [Critical factors for mission success]"""
        
        try:
            response = self.llm_assistant.llm.invoke(briefing_prompt)
            enhanced_text = response.content if hasattr(response, 'content') else str(response)
            return enhanced_text
        except Exception as e:
            print(f"[LLM BRIEFING] Error: {e}")
            return basic_briefing
    
    def evaluate_mission_performance(self, mission: Mission, episode_data: Dict[str, Any],
                                   agent_actions: List[np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate agent performance against mission objectives
        
        Args:
            mission: Mission definition
            episode_data: Episode performance data
            agent_actions: Sequence of agent actions
        
        Returns:
            Mission evaluation results
        """
        
        evaluation = {
            'mission_id': mission.mission_id,
            'mission_type': mission.mission_type.value,
            'objectives_evaluation': {},
            'overall_success': False,
            'performance_score': 0.0,
            'tactical_assessment': {},
            'recommendations': []
        }
        
        # Evaluate each objective
        total_objective_score = 0.0
        objectives_met = 0
        
        for objective in mission.objectives:
            obj_success = self._evaluate_objective(objective, episode_data)
            evaluation['objectives_evaluation'][objective.type] = obj_success
            
            if obj_success['success']:
                objectives_met += 1
                if objective.priority == 'primary':
                    total_objective_score += 1.0
                elif objective.priority == 'secondary':
                    total_objective_score += 0.5
                else:
                    total_objective_score += 0.2
        
        # Overall mission success
        primary_objectives = [obj for obj in mission.objectives if obj.priority == 'primary']
        primary_success_rate = sum(1 for obj in primary_objectives 
                                 if evaluation['objectives_evaluation'][obj.type]['success']) / len(primary_objectives)
        
        evaluation['overall_success'] = primary_success_rate >= mission.success_criteria.get('primary_objectives_completed', 1.0)
        evaluation['performance_score'] = total_objective_score / len(mission.objectives)
        
        # Tactical assessment
        evaluation['tactical_assessment'] = self._assess_tactical_performance(
            mission, episode_data, agent_actions
        )
        
        # Generate recommendations
        evaluation['recommendations'] = self._generate_performance_recommendations(
            mission, evaluation
        )
        
        if self.verbose:
            print(f"[MISSION EVAL] {mission.mission_id}: Success={evaluation['overall_success']}, "
                  f"Score={evaluation['performance_score']:.2f}")
        
        return evaluation
    
    def _evaluate_objective(self, objective: MissionObjective, 
                          episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate individual objective completion"""
        
        success = False
        details = {}
        
        if objective.type == "destroy_enemy_aircraft":
            enemy_destroyed = episode_data.get('enemy_health', 1.0) <= 0
            own_survival = episode_data.get('ego_health', 0.0) > 0
            success = enemy_destroyed and own_survival
            details = {'enemy_destroyed': enemy_destroyed, 'survival': own_survival}
        
        elif objective.type == "intercept_target":
            # Check if intercept was successful within time limit
            mission_time = episode_data.get('episode_length', 0)
            time_limit = objective.time_limit or 600
            success = (episode_data.get('enemy_health', 1.0) <= 0 and 
                      mission_time <= time_limit)
            details = {'time_used': mission_time, 'time_limit': time_limit}
        
        elif objective.type == "area_patrol":
            # Check patrol duration
            patrol_time = episode_data.get('episode_length', 0)
            required_time = objective.success_criteria.get('patrol_time', 1800)
            success = patrol_time >= required_time
            details = {'patrol_duration': patrol_time, 'required_duration': required_time}
        
        return {
            'success': success,
            'details': details,
            'objective_type': objective.type,
            'priority': objective.priority
        }
    
    def _assess_tactical_performance(self, mission: Mission, episode_data: Dict[str, Any],
                                   agent_actions: List[np.ndarray]) -> Dict[str, Any]:
        """Assess tactical performance during mission"""
        
        # Basic tactical metrics
        tactical_metrics = {
            'action_smoothness': self._calculate_action_smoothness(agent_actions),
            'energy_management': episode_data.get('energy_violations', 0),
            'threat_response': episode_data.get('defensive_actions', 0),
            'weapon_efficiency': episode_data.get('shot_accuracy', 0),
            'situational_awareness': episode_data.get('lock_acquisition_rate', 0)
        }
        
        # Mission-specific assessments
        if mission.mission_type == MissionType.AIR_SUPERIORITY:
            tactical_metrics['air_superiority_tactics'] = self._assess_air_superiority_tactics(episode_data)
        elif mission.mission_type == MissionType.INTERCEPT:
            tactical_metrics['intercept_efficiency'] = self._assess_intercept_efficiency(episode_data)
        
        return tactical_metrics
    
    def _calculate_action_smoothness(self, actions: List[np.ndarray]) -> float:
        """Calculate smoothness of control inputs"""
        if len(actions) < 2:
            return 1.0
        
        # Calculate action changes
        action_changes = []
        for i in range(1, len(actions)):
            change = np.linalg.norm(actions[i][:3] - actions[i-1][:3])  # Control inputs only
            action_changes.append(change)
        
        # Smoothness = 1 / (1 + average_change)
        avg_change = np.mean(action_changes)
        smoothness = 1.0 / (1.0 + avg_change)
        
        return smoothness
    
    def _assess_air_superiority_tactics(self, episode_data: Dict[str, Any]) -> float:
        """Assess air superiority specific tactics"""
        score = 0.0
        
        # BVR engagement effectiveness
        if episode_data.get('bvr_engagements', 0) > 0:
            bvr_success = episode_data.get('bvr_success_rate', 0)
            score += bvr_success * 0.4
        
        # Energy management
        energy_score = 1.0 - min(1.0, episode_data.get('energy_violations', 0) / 10.0)
        score += energy_score * 0.3
        
        # Overall engagement success
        if episode_data.get('enemy_health', 1.0) <= 0:
            score += 0.3
        
        return min(1.0, score)
    
    def _assess_intercept_efficiency(self, episode_data: Dict[str, Any]) -> float:
        """Assess intercept mission efficiency"""
        
        # Time efficiency
        mission_time = episode_data.get('episode_length', 0)
        optimal_time = 300  # 5 minutes optimal intercept
        time_efficiency = max(0.0, 1.0 - (mission_time - optimal_time) / optimal_time)
        
        # Intercept success
        intercept_success = 1.0 if episode_data.get('enemy_health', 1.0) <= 0 else 0.0
        
        return (time_efficiency * 0.4 + intercept_success * 0.6)
    
    def _generate_performance_recommendations(self, mission: Mission, 
                                            evaluation: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        # Overall performance recommendations
        if evaluation['performance_score'] < 0.5:
            recommendations.append("Focus on primary objective completion")
        
        # Tactical recommendations
        tactical_assessment = evaluation.get('tactical_assessment', {})
        
        if tactical_assessment.get('action_smoothness', 1.0) < 0.6:
            recommendations.append("Improve control input smoothness - avoid excessive maneuvering")
        
        if tactical_assessment.get('weapon_efficiency', 1.0) < 0.4:
            recommendations.append("Improve weapon employment - ensure proper lock before firing")
        
        if tactical_assessment.get('energy_management', 0) > 5:
            recommendations.append("Focus on energy management - maintain altitude and speed advantage")
        
        # Mission-specific recommendations
        if mission.mission_type == MissionType.AIR_SUPERIORITY:
            if tactical_assessment.get('air_superiority_tactics', 0) < 0.5:
                recommendations.append("Study BVR engagement tactics - maintain standoff distance")
        
        elif mission.mission_type == MissionType.INTERCEPT:
            if tactical_assessment.get('intercept_efficiency', 0) < 0.6:
                recommendations.append("Improve intercept geometry - optimize approach angle")
        
        return recommendations
    
    def get_mission_statistics(self) -> Dict[str, Any]:
        """Get mission generation and performance statistics"""
        
        if not self.mission_history:
            return {'missions_generated': 0}
        
        # Analyze mission history
        mission_types = [m['type'] for m in self.mission_history]
        type_counts = {mt.value: mission_types.count(mt.value) for mt in MissionType}
        
        difficulties = [m['difficulty'] for m in self.mission_history]
        difficulty_counts = {d: difficulties.count(d) for d in ['easy', 'medium', 'hard', 'expert']}
        
        stats = {
            'missions_generated': len(self.mission_history),
            'mission_type_distribution': type_counts,
            'difficulty_distribution': difficulty_counts,
            'generation_rate': len(self.mission_history) / max(1, time.time() - self.mission_history[0]['generation_time']) * 3600,  # per hour
            'most_common_type': max(type_counts, key=type_counts.get),
            'average_complexity': np.mean([1 if d == 'easy' else 2 if d == 'medium' else 3 if d == 'hard' else 4 for d in difficulties])
        }
        
        return stats


class MissionBasedTrainingManager:
    """
    Training manager that integrates mission planning with RL training
    """
    
    def __init__(self, mission_planner: MissionPlanner, curriculum_manager=None):
        """
        Initialize mission-based training manager
        
        Args:
            mission_planner: Mission planner for generating scenarios
            curriculum_manager: Optional curriculum manager for progressive training
        """
        self.mission_planner = mission_planner
        self.curriculum_manager = curriculum_manager
        
        # Training state
        self.current_mission = None
        self.mission_results = []
        self.training_metrics = {}
        
        print("[MISSION TRAINING] Mission-based training manager initialized")
    
    def generate_training_mission(self, stage_config: Dict[str, Any] = None) -> Mission:
        """Generate mission appropriate for current training stage"""
        
        # Use curriculum stage if available
        if self.curriculum_manager and stage_config is None:
            current_stage = self.curriculum_manager.get_current_stage()
            mission_type = self._select_mission_for_stage(current_stage.name)
            difficulty = current_stage.scenario_complexity
        else:
            # Random mission generation
            mission_type = random.choice(list(MissionType))
            difficulty = stage_config.get('difficulty', 'medium') if stage_config else 'medium'
        
        # Generate mission
        mission = self.mission_planner.generate_mission(
            mission_type=mission_type,
            difficulty=difficulty,
            custom_params=stage_config
        )
        
        self.current_mission = mission
        return mission
    
    def _select_mission_for_stage(self, stage_name: str) -> MissionType:
        """Select appropriate mission type for curriculum stage"""
        
        stage_mission_mapping = {
            'basic_flight': MissionType.CAP,  # Simple patrol
            'target_acquisition': MissionType.INTERCEPT,  # Focus on targeting
            'bvr_engagement': MissionType.AIR_SUPERIORITY,  # BVR focus
            'wvr_dogfight': MissionType.AIR_SUPERIORITY,  # WVR focus
            'defensive_tactics': MissionType.ESCORT,  # Defensive focus
            'advanced_combat': MissionType.AIR_SUPERIORITY,  # Complex scenarios
            'ace_level': random.choice(list(MissionType))  # All mission types
        }
        
        return stage_mission_mapping.get(stage_name, MissionType.AIR_SUPERIORITY)
    
    def evaluate_mission_completion(self, episode_data: Dict[str, Any],
                                  agent_actions: List[np.ndarray]) -> Dict[str, Any]:
        """Evaluate completion of current mission"""
        
        if self.current_mission is None:
            return {'error': 'No current mission'}
        
        # Evaluate mission performance
        evaluation = self.mission_planner.evaluate_mission_performance(
            self.current_mission, episode_data, agent_actions
        )
        
        # Store results
        self.mission_results.append(evaluation)
        
        # Update training metrics
        self._update_training_metrics(evaluation)
        
        return evaluation
    
    def _update_training_metrics(self, evaluation: Dict[str, Any]):
        """Update overall training metrics based on mission evaluation"""
        
        mission_type = evaluation['mission_type']
        
        # Initialize mission type metrics if not exists
        if mission_type not in self.training_metrics:
            self.training_metrics[mission_type] = {
                'missions_attempted': 0,
                'missions_successful': 0,
                'average_score': 0.0,
                'performance_history': []
            }
        
        # Update metrics
        metrics = self.training_metrics[mission_type]
        metrics['missions_attempted'] += 1
        
        if evaluation['overall_success']:
            metrics['missions_successful'] += 1
        
        # Update average score
        prev_avg = metrics['average_score']
        new_score = evaluation['performance_score']
        n = metrics['missions_attempted']
        metrics['average_score'] = (prev_avg * (n-1) + new_score) / n
        
        # Store performance history
        metrics['performance_history'].append({
            'mission_id': evaluation['mission_id'],
            'success': evaluation['overall_success'],
            'score': evaluation['performance_score'],
            'timestamp': time.time()
        })
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary across all missions"""
        
        summary = {
            'total_missions': len(self.mission_results),
            'overall_success_rate': 0.0,
            'mission_type_performance': {},
            'curriculum_progress': None
        }
        
        if self.mission_results:
            # Calculate overall success rate
            successful_missions = sum(1 for result in self.mission_results if result['overall_success'])
            summary['overall_success_rate'] = successful_missions / len(self.mission_results)
            
            # Mission type performance
            for mission_type, metrics in self.training_metrics.items():
                summary['mission_type_performance'][mission_type] = {
                    'success_rate': metrics['missions_successful'] / metrics['missions_attempted'],
                    'average_score': metrics['average_score'],
                    'missions_completed': metrics['missions_attempted']
                }
        
        # Curriculum progress if available
        if self.curriculum_manager:
            summary['curriculum_progress'] = self.curriculum_manager.get_curriculum_summary()
        
        return summary


if __name__ == "__main__":
    # Example usage
    print("Mission Planning System for Context-Aware Combat Training")
    
    # Create mission planner
    planner = MissionPlanner(verbose=True)
    
    # Generate example missions
    for mission_type in [MissionType.AIR_SUPERIORITY, MissionType.INTERCEPT, MissionType.ESCORT]:
        mission = planner.generate_mission(mission_type, difficulty="medium")
        print(f"\nGenerated {mission_type.value} mission:")
        print(f"  Objectives: {len(mission.objectives)}")
        print(f"  Threat Level: {mission.threat_assessment.overall_threat_level.value}")
        print(f"  Air Threats: {len(mission.threat_assessment.air_threats)}")
    
    # Print statistics
    stats = planner.get_mission_statistics()
    print(f"\nMission Statistics: {stats}")
