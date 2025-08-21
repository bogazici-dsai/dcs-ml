# Multi-Stage LLM Tactical Assistant for Advanced Combat Reasoning
import json
import time
import math
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class ReasoningStage(Enum):
    """Different stages of tactical reasoning"""
    STRATEGIC = "strategic"      # Mission-level planning (every 100 steps)
    TACTICAL = "tactical"        # Engagement-level decisions (every 20 steps)
    EXECUTION = "execution"      # Action-level critique (every step)


@dataclass
class TacticalContext:
    """Context for tactical decision making"""
    mission_type: str = "air_superiority"
    threat_environment: str = "medium"
    rules_of_engagement: str = "weapons_tight"
    time_pressure: str = "medium"
    fuel_state: str = "normal"
    ammunition_state: str = "full"


class MultiStageTacticalAssistant:
    """
    Advanced tactical assistant with hierarchical reasoning:
    1. Strategic Planning (mission-level, long-term)
    2. Tactical Assessment (engagement-level, medium-term) 
    3. Execution Critique (action-level, immediate)
    """
    
    def __init__(self, llm, verbose: bool = True, max_rate_hz: float = 10.0):
        """
        Initialize multi-stage tactical assistant
        
        Args:
            llm: Language model for reasoning
            verbose: Enable detailed logging
            max_rate_hz: Maximum LLM call rate
        """
        self.llm = llm
        self.verbose = verbose
        self.max_rate_hz = max_rate_hz
        
        # Stage-specific rate limiting
        self.stage_timers = {
            ReasoningStage.STRATEGIC: 0.0,
            ReasoningStage.TACTICAL: 0.0,
            ReasoningStage.EXECUTION: 0.0
        }
        
        # Stage-specific intervals (in seconds)
        self.stage_intervals = {
            ReasoningStage.STRATEGIC: 10.0,   # Every 10 seconds
            ReasoningStage.TACTICAL: 2.0,     # Every 2 seconds
            ReasoningStage.EXECUTION: 0.1     # Every 0.1 seconds (10 Hz)
        }
        
        # Cached reasoning results
        self.strategic_plan = None
        self.tactical_assessment = None
        self.execution_feedback = None
        
        # Performance tracking
        self.reasoning_history = {
            ReasoningStage.STRATEGIC: [],
            ReasoningStage.TACTICAL: [],
            ReasoningStage.EXECUTION: []
        }
        
        # Mission context
        self.mission_context = TacticalContext()
        
        print(f"[MULTI STAGE] Assistant initialized with hierarchical reasoning")
        print(f"[MULTI STAGE] Strategic: {self.stage_intervals[ReasoningStage.STRATEGIC]}s intervals")
        print(f"[MULTI STAGE] Tactical: {self.stage_intervals[ReasoningStage.TACTICAL]}s intervals")
        print(f"[MULTI STAGE] Execution: {self.stage_intervals[ReasoningStage.EXECUTION]}s intervals")
    
    def get_comprehensive_guidance(self, features: Dict[str, Any], 
                                  step: int) -> Tuple[float, Dict[str, Any]]:
        """
        Get comprehensive guidance from all reasoning stages
        
        Args:
            features: Tactical features from environment
            step: Current step number
        
        Returns:
            (shaping_delta, comprehensive_response)
        """
        now = time.time()
        comprehensive_response = {
            'strategic': self.strategic_plan,
            'tactical': self.tactical_assessment,
            'execution': self.execution_feedback,
            'combined_shaping': 0.0,
            'primary_guidance': 'execution',
            'reasoning_stages_used': []
        }
        
        # Stage 1: Strategic Planning (least frequent, highest level)
        if self._should_update_stage(ReasoningStage.STRATEGIC, now):
            self.strategic_plan = self._strategic_planning(features, step)
            comprehensive_response['strategic'] = self.strategic_plan
            comprehensive_response['reasoning_stages_used'].append('strategic')
            
            if self.verbose:
                print(f"[STRATEGIC] Updated plan: {self.strategic_plan.get('mission_phase', 'unknown')}")
        
        # Stage 2: Tactical Assessment (medium frequency, engagement level)
        if self._should_update_stage(ReasoningStage.TACTICAL, now):
            self.tactical_assessment = self._tactical_assessment(features, self.strategic_plan)
            comprehensive_response['tactical'] = self.tactical_assessment
            comprehensive_response['reasoning_stages_used'].append('tactical')
            
            if self.verbose:
                print(f"[TACTICAL] Updated assessment: {self.tactical_assessment.get('engagement_priority', 'unknown')}")
        
        # Stage 3: Execution Critique (highest frequency, immediate feedback)
        if self._should_update_stage(ReasoningStage.EXECUTION, now):
            self.execution_feedback = self._execution_critique(features, self.tactical_assessment)
            comprehensive_response['execution'] = self.execution_feedback
            comprehensive_response['reasoning_stages_used'].append('execution')
        
        # Combine all guidance into final shaping delta
        combined_shaping = self._combine_multi_stage_guidance(
            strategic=self.strategic_plan,
            tactical=self.tactical_assessment, 
            execution=self.execution_feedback,
            features=features
        )
        
        comprehensive_response['combined_shaping'] = combined_shaping
        
        return combined_shaping, comprehensive_response
    
    def _should_update_stage(self, stage: ReasoningStage, current_time: float) -> bool:
        """Check if a reasoning stage should be updated based on timing"""
        last_update = self.stage_timers[stage]
        interval = self.stage_intervals[stage]
        
        if current_time - last_update >= interval:
            self.stage_timers[stage] = current_time
            return True
        return False
    
    def _strategic_planning(self, features: Dict[str, Any], step: int) -> Dict[str, Any]:
        """
        Strategic-level mission planning and long-term objectives
        
        Focuses on:
        - Mission phase assessment
        - Long-term positioning strategy
        - Resource management
        - Risk assessment
        """
        
        distance = features.get('distance', 0)
        energy_state = features.get('energy_state', 'MEDIUM')
        threat_level = features.get('threat_level', 0)
        engagement_phase = features.get('engagement_phase', 'UNKNOWN')
        
        strategic_prompt = f"""You are an experienced fighter squadron commander providing strategic guidance.

=== MISSION CONTEXT ===
Mission Type: {self.mission_context.mission_type}
Threat Environment: {self.mission_context.threat_environment}
Rules of Engagement: {self.mission_context.rules_of_engagement}
Fuel State: {self.mission_context.fuel_state}
Ammunition: {self.mission_context.ammunition_state}

=== CURRENT STRATEGIC PICTURE ===
Step: {step}
Range to Target: {distance:.0f}m
Energy State: {energy_state}
Threat Level: {threat_level:.2f}
Current Phase: {engagement_phase}

=== STRATEGIC PLANNING REQUIREMENTS ===
Analyze the mission from a strategic perspective and provide:

1. Mission Phase Assessment (APPROACH/ENGAGE/DISENGAGE/RTB)
2. Long-term Positioning Strategy
3. Resource Management Priorities
4. Risk Assessment and Mitigation

Respond in JSON format:
{{
    "mission_phase": "<current mission phase>",
    "strategic_priority": "<primary strategic objective>",
    "positioning_strategy": "<long-term positioning plan>",
    "resource_management": "<fuel/ammo conservation strategy>",
    "risk_assessment": "<primary risks and mitigation>",
    "success_probability": <float 0-1>,
    "recommended_engagement_strategy": "<overall engagement approach>",
    "strategic_shaping": <float -0.2 to 0.2>
}}"""
        
        try:
            response = self.llm.invoke(strategic_prompt)
            text = response.content if hasattr(response, 'content') else str(response)
            strategic_data = json.loads(text)
            
            # Validate and store
            strategic_data['stage'] = 'strategic'
            strategic_data['timestamp'] = time.time()
            self.reasoning_history[ReasoningStage.STRATEGIC].append(strategic_data)
            
            return strategic_data
            
        except Exception as e:
            if self.verbose:
                print(f"[STRATEGIC ERROR] {e}")
            return {
                'mission_phase': 'ENGAGE',
                'strategic_priority': 'NEUTRALIZE_THREAT',
                'strategic_shaping': 0.0,
                'error': str(e)
            }
    
    def _tactical_assessment(self, features: Dict[str, Any], 
                           strategic_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tactical-level engagement assessment and medium-term planning
        
        Focuses on:
        - Engagement geometry optimization
        - Immediate threat response
        - Weapons employment decisions
        - Maneuvering priorities
        """
        
        distance = features.get('distance', 0)
        aspect_angle = features.get('aspect_angle', 0)
        closure_rate = features.get('closure_rate', 0)
        locked = features.get('locked', -1)
        threat_level = features.get('threat_level', 0)
        pursuit_geometry = features.get('pursuit_geometry', 'NEUTRAL')
        missile_zone = features.get('missile_employment_zone', 'NO_LOCK')
        
        strategic_priority = strategic_plan.get('strategic_priority', 'ENGAGE') if strategic_plan else 'ENGAGE'
        
        tactical_prompt = f"""You are a fighter pilot weapons systems officer providing tactical engagement guidance.

=== STRATEGIC CONTEXT ===
Mission Phase: {strategic_plan.get('mission_phase', 'ENGAGE') if strategic_plan else 'ENGAGE'}
Strategic Priority: {strategic_priority}

=== TACTICAL PICTURE ===
Range: {distance:.0f}m | Aspect: {aspect_angle:.1f}° | Closure: {closure_rate:.1f}m/s
Lock Status: {'LOCKED' if locked > 0 else 'NO LOCK'}
Threat Level: {threat_level:.2f}
Pursuit Geometry: {pursuit_geometry}
Weapon Employment Zone: {missile_zone}

=== TACTICAL ASSESSMENT REQUIREMENTS ===
Provide immediate tactical guidance for this engagement:

1. Engagement Priority (OFFENSIVE/DEFENSIVE/NEUTRAL)
2. Optimal Maneuvering Strategy
3. Weapons Employment Recommendation
4. Immediate Threat Response

Respond in JSON format:
{{
    "engagement_priority": "<OFFENSIVE/DEFENSIVE/NEUTRAL>",
    "maneuvering_strategy": "<optimal maneuvering approach>",
    "weapons_employment": "<weapons recommendation>",
    "threat_response": "<immediate threat mitigation>",
    "geometric_advantage": "<current geometric position assessment>",
    "recommended_actions": ["<list of recommended actions>"],
    "tactical_shaping": <float -0.3 to 0.3>
}}"""
        
        try:
            response = self.llm.invoke(tactical_prompt)
            text = response.content if hasattr(response, 'content') else str(response)
            tactical_data = json.loads(text)
            
            # Validate and store
            tactical_data['stage'] = 'tactical'
            tactical_data['timestamp'] = time.time()
            self.reasoning_history[ReasoningStage.TACTICAL].append(tactical_data)
            
            return tactical_data
            
        except Exception as e:
            if self.verbose:
                print(f"[TACTICAL ERROR] {e}")
            return {
                'engagement_priority': 'DEFENSIVE',
                'maneuvering_strategy': 'MAINTAIN_DISTANCE',
                'tactical_shaping': 0.0,
                'error': str(e)
            }
    
    def _execution_critique(self, features: Dict[str, Any],
                          tactical_assessment: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execution-level action critique and immediate feedback
        
        Focuses on:
        - Control input quality
        - Immediate action effectiveness
        - Micro-adjustments
        - Safety concerns
        """
        
        action = features.get('action', [0, 0, 0, 0])
        action_smoothness = features.get('action_smoothness', 0)
        g_force = features.get('g_force', 0)
        turn_rate = features.get('turn_rate', 0)
        in_firing_envelope = features.get('in_firing_envelope', False)
        defensive_urgency = features.get('defensive_urgency', 'NEUTRAL')
        
        tactical_priority = tactical_assessment.get('engagement_priority', 'NEUTRAL') if tactical_assessment else 'NEUTRAL'
        
        execution_prompt = f"""You are a fighter pilot instructor providing immediate action feedback.

=== TACTICAL CONTEXT ===
Current Priority: {tactical_priority}
Defensive Urgency: {defensive_urgency}
In Firing Envelope: {'YES' if in_firing_envelope else 'NO'}

=== CURRENT ACTION ANALYSIS ===
Control Inputs: Pitch={action[0]:.2f}, Roll={action[1]:.2f}, Yaw={action[2]:.2f}, Fire={action[3]:.2f}
Action Smoothness: {action_smoothness:.3f} (lower = smoother)
G-Force: {g_force:.1f}G
Turn Rate: {turn_rate:.1f}°/s

=== EXECUTION CRITIQUE REQUIREMENTS ===
Provide immediate feedback on the current action:

1. Control Input Quality Assessment
2. Action Effectiveness for Current Situation
3. Safety Concerns (G-limits, structural stress)
4. Micro-adjustments Needed

Respond in JSON format:
{{
    "control_quality": "<assessment of control inputs>",
    "action_effectiveness": "<how well action fits situation>", 
    "safety_assessment": "<any safety concerns>",
    "micro_adjustments": "<specific input adjustments needed>",
    "execution_score": <float 0-1>,
    "execution_shaping": <float -0.1 to 0.1>
}}

Keep response concise and focused on immediate action quality."""
        
        try:
            response = self.llm.invoke(execution_prompt)
            text = response.content if hasattr(response, 'content') else str(response)
            execution_data = json.loads(text)
            
            # Validate and store
            execution_data['stage'] = 'execution'
            execution_data['timestamp'] = time.time()
            self.reasoning_history[ReasoningStage.EXECUTION].append(execution_data)
            
            return execution_data
            
        except Exception as e:
            if self.verbose:
                print(f"[EXECUTION ERROR] {e}")
            return {
                'control_quality': 'UNKNOWN',
                'action_effectiveness': 'UNKNOWN',
                'execution_shaping': 0.0,
                'error': str(e)
            }
    
    def _combine_multi_stage_guidance(self, strategic: Optional[Dict[str, Any]],
                                    tactical: Optional[Dict[str, Any]],
                                    execution: Optional[Dict[str, Any]],
                                    features: Dict[str, Any]) -> float:
        """
        Combine guidance from all reasoning stages into final shaping delta
        
        Weighting strategy:
        - Strategic: 0.3 weight (long-term mission success)
        - Tactical: 0.4 weight (engagement effectiveness)  
        - Execution: 0.3 weight (immediate action quality)
        """
        
        # Extract shaping deltas from each stage
        strategic_shaping = strategic.get('strategic_shaping', 0.0) if strategic else 0.0
        tactical_shaping = tactical.get('tactical_shaping', 0.0) if tactical else 0.0
        execution_shaping = execution.get('execution_shaping', 0.0) if execution else 0.0
        
        # Adaptive weighting based on situation urgency
        threat_level = features.get('threat_level', 0)
        defensive_urgency = features.get('defensive_urgency', 'NEUTRAL')
        
        if threat_level > 0.8 or defensive_urgency == 'BREAK_BREAK':
            # Emergency situation - prioritize execution and tactical
            weights = {'strategic': 0.1, 'tactical': 0.5, 'execution': 0.4}
        elif threat_level > 0.5 or defensive_urgency == 'DEFENSIVE':
            # High threat - balance tactical and execution
            weights = {'strategic': 0.2, 'tactical': 0.5, 'execution': 0.3}
        else:
            # Normal situation - balanced approach
            weights = {'strategic': 0.3, 'tactical': 0.4, 'execution': 0.3}
        
        # Combine with adaptive weights
        combined_shaping = (
            strategic_shaping * weights['strategic'] +
            tactical_shaping * weights['tactical'] +
            execution_shaping * weights['execution']
        )
        
        # Safety clamp
        combined_shaping = max(-0.5, min(0.5, combined_shaping))
        
        if self.verbose and abs(combined_shaping) > 0.1:
            print(f"[MULTI STAGE] Combined shaping: {combined_shaping:.3f} "
                  f"(S:{strategic_shaping:.2f}, T:{tactical_shaping:.2f}, E:{execution_shaping:.2f})")
        
        return combined_shaping
    
    def update_mission_context(self, mission_type: str = None, threat_environment: str = None,
                             fuel_state: str = None, ammunition_state: str = None):
        """Update mission context for strategic planning"""
        if mission_type:
            self.mission_context.mission_type = mission_type
        if threat_environment:
            self.mission_context.threat_environment = threat_environment
        if fuel_state:
            self.mission_context.fuel_state = fuel_state
        if ammunition_state:
            self.mission_context.ammunition_state = ammunition_state
        
        # Clear strategic plan to force re-evaluation
        self.strategic_plan = None
        
        if self.verbose:
            print(f"[CONTEXT] Mission context updated: {mission_type}, {threat_environment}")
    
    def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get summary of multi-stage reasoning performance"""
        summary = {
            'total_strategic_plans': len(self.reasoning_history[ReasoningStage.STRATEGIC]),
            'total_tactical_assessments': len(self.reasoning_history[ReasoningStage.TACTICAL]),
            'total_execution_critiques': len(self.reasoning_history[ReasoningStage.EXECUTION]),
            'current_mission_context': self.mission_context.__dict__,
            'stage_intervals': {stage.value: interval for stage, interval in self.stage_intervals.items()},
            'last_strategic_plan': self.strategic_plan,
            'last_tactical_assessment': self.tactical_assessment,
            'last_execution_feedback': self.execution_feedback
        }
        
        # Calculate average shaping deltas per stage
        for stage in ReasoningStage:
            history = self.reasoning_history[stage]
            if history:
                shaping_key = f"{stage.value}_shaping"
                shapings = [item.get(shaping_key, 0) for item in history if shaping_key in item]
                if shapings:
                    summary[f'avg_{stage.value}_shaping'] = np.mean(shapings)
                    summary[f'std_{stage.value}_shaping'] = np.std(shapings)
        
        return summary
    
    def export_reasoning_history(self, filepath: str):
        """Export complete reasoning history for analysis"""
        export_data = {
            'mission_context': self.mission_context.__dict__,
            'stage_intervals': {stage.value: interval for stage, interval in self.stage_intervals.items()},
            'reasoning_history': {
                stage.value: history for stage, history in self.reasoning_history.items()
            },
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"[EXPORT] Reasoning history exported to: {filepath}")


class EnhancedTacticalAssistant:
    """
    Enhanced version of HarfangTacticalAssistant with multi-stage reasoning
    """
    
    def __init__(self, llm, verbose: bool = True, max_rate_hz: float = 10.0,
                 use_multi_stage: bool = True):
        """
        Initialize enhanced tactical assistant
        
        Args:
            llm: Language model
            verbose: Verbose output
            max_rate_hz: Maximum call rate
            use_multi_stage: Whether to use multi-stage reasoning
        """
        # Initialize base tactical assistant
        from HarfangAssistant_Enhanced import HarfangTacticalAssistant
        self.base_assistant = HarfangTacticalAssistant(llm, verbose, max_rate_hz)
        
        # Add multi-stage reasoning if requested
        self.use_multi_stage = use_multi_stage
        if use_multi_stage:
            self.multi_stage_assistant = MultiStageTacticalAssistant(llm, verbose, max_rate_hz)
        
        self.verbose = verbose
        self.max_rate_hz = max_rate_hz  # Store for compatibility
        
        print(f"[ENHANCED ASSISTANT] Initialized with multi-stage: {use_multi_stage}")
    
    def request_shaping(self, features: Dict[str, Any], step: int = 0) -> Tuple[float, Dict[str, Any]]:
        """
        Request tactical shaping with optional multi-stage reasoning
        
        Args:
            features: Tactical features
            step: Current step number
        
        Returns:
            (shaping_delta, response_data)
        """
        if self.use_multi_stage:
            # Use multi-stage reasoning
            shaping_delta, response_data = self.multi_stage_assistant.get_comprehensive_guidance(features, step)
            
            # Add base assistant features for compatibility
            response_data['critique'] = response_data.get('execution', {}).get('control_quality', 'Multi-stage guidance')
            response_data['shaping_delta'] = shaping_delta
            
            return shaping_delta, response_data
        else:
            # Use base assistant
            return self.base_assistant.request_shaping(features)
    
    def extract_features(self, *args, **kwargs):
        """Extract features using base assistant"""
        return self.base_assistant.extract_features(*args, **kwargs)
    
    def update_mission_context(self, **kwargs):
        """Update mission context if using multi-stage reasoning"""
        if self.use_multi_stage:
            self.multi_stage_assistant.update_mission_context(**kwargs)
    
    def get_tactical_summary(self) -> Dict[str, Any]:
        """Get comprehensive tactical summary"""
        base_summary = self.base_assistant.get_tactical_summary()
        
        if self.use_multi_stage:
            multi_stage_summary = self.multi_stage_assistant.get_reasoning_summary()
            base_summary.update(multi_stage_summary)
        
        return base_summary


if __name__ == "__main__":
    print("Multi-Stage Tactical Assistant for Advanced Combat Reasoning")
    print("Provides Strategic → Tactical → Execution level guidance")
    print("Usage: Import and use with enhanced RL training system")
