# Hierarchical Action Space for Advanced Combat Maneuvers
import numpy as np
import math
import json
import time
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque
import gymnasium as gym


class MacroActionType(Enum):
    """Types of high-level tactical maneuvers"""
    # Defensive maneuvers
    DEFENSIVE_SPIRAL = "defensive_spiral"
    BARREL_ROLL = "barrel_roll"
    SPLIT_S = "split_s"
    CHANDELLE = "chandelle"
    NOTCH_MANEUVER = "notch_maneuver"
    
    # Offensive maneuvers
    IMMELMANN_TURN = "immelmann_turn"
    HIGH_YO_YO = "high_yo_yo"
    LOW_YO_YO = "low_yo_yo"
    LEAD_PURSUIT = "lead_pursuit"
    PURE_PURSUIT = "pure_pursuit"
    
    # Energy maneuvers
    ZOOM_CLIMB = "zoom_climb"
    ENERGY_DIVE = "energy_dive"
    LEVEL_ACCELERATION = "level_acceleration"
    
    # Tactical maneuvers
    CRANK_MANEUVER = "crank_maneuver"
    BEAM_MANEUVER = "beam_maneuver"
    DRAG_MANEUVER = "drag_maneuver"


@dataclass
class MacroActionSequence:
    """Definition of a macro action sequence"""
    name: str
    description: str
    duration_steps: int
    control_sequence: List[np.ndarray]  # Sequence of [pitch, roll, yaw, fire] actions
    prerequisites: Dict[str, Any]       # Conditions required to execute
    effectiveness_conditions: Dict[str, Any]  # When this maneuver is most effective
    energy_cost: float                  # Energy cost (0-1)
    risk_level: float                   # Risk level (0-1)


class MacroActionLibrary:
    """
    Library of predefined tactical maneuvers based on real fighter pilot techniques
    """
    
    def __init__(self):
        """Initialize macro action library with realistic combat maneuvers"""
        
        self.macro_actions = {}
        self._initialize_defensive_maneuvers()
        self._initialize_offensive_maneuvers()
        self._initialize_energy_maneuvers()
        self._initialize_tactical_maneuvers()
        
        print(f"[MACRO LIBRARY] Initialized with {len(self.macro_actions)} macro actions")
        self._print_available_maneuvers()
    
    def _initialize_defensive_maneuvers(self):
        """Initialize defensive Basic Fighter Maneuvers (BFM)"""
        
        # Defensive Spiral - Classic defensive maneuver
        defensive_spiral_sequence = []
        for step in range(30):  # 30-step sequence
            # Descending spiral with increasing turn rate
            pitch = -0.3 - (step * 0.02)  # Increasing dive
            roll = 0.8 * math.sin(step * 0.3)  # Rolling motion
            yaw = 0.4 * math.cos(step * 0.2)   # Coordinated turn
            fire = 0.0  # No firing during defensive maneuver
            
            defensive_spiral_sequence.append(np.array([pitch, roll, yaw, fire]))
        
        self.macro_actions[MacroActionType.DEFENSIVE_SPIRAL] = MacroActionSequence(
            name="Defensive Spiral",
            description="Descending spiral to defeat missile or gain separation",
            duration_steps=30,
            control_sequence=defensive_spiral_sequence,
            prerequisites={"threat_level": 0.6, "energy_state": "MEDIUM"},
            effectiveness_conditions={"under_attack": True, "missile_incoming": True},
            energy_cost=0.7,  # High energy cost
            risk_level=0.4    # Medium risk
        )
        
        # Barrel Roll - Defensive displacement maneuver
        barrel_roll_sequence = []
        for step in range(20):
            # Complete barrel roll over 20 steps
            angle = (step / 20.0) * 2 * math.pi
            pitch = 0.2 * math.sin(angle)
            roll = 0.9 * math.cos(angle)
            yaw = 0.1 * math.sin(angle * 2)
            fire = 0.0
            
            barrel_roll_sequence.append(np.array([pitch, roll, yaw, fire]))
        
        self.macro_actions[MacroActionType.BARREL_ROLL] = MacroActionSequence(
            name="Barrel Roll",
            description="Displacement roll to avoid gunfire or gain position",
            duration_steps=20,
            control_sequence=barrel_roll_sequence,
            prerequisites={"energy_state": "MEDIUM", "distance": 3000},
            effectiveness_conditions={"under_guns_attack": True, "wvr_combat": True},
            energy_cost=0.5,
            risk_level=0.3
        )
        
        # Split-S - Emergency defensive maneuver
        split_s_sequence = []
        for step in range(15):
            if step < 5:  # Initial roll inverted
                pitch = 0.0
                roll = 0.9
                yaw = 0.0
            else:  # Pull through to dive
                pitch = -0.8
                roll = 0.0
                yaw = 0.0
            fire = 0.0
            
            split_s_sequence.append(np.array([pitch, roll, yaw, fire]))
        
        self.macro_actions[MacroActionType.SPLIT_S] = MacroActionSequence(
            name="Split-S",
            description="Emergency maneuver to quickly reverse direction and dive",
            duration_steps=15,
            control_sequence=split_s_sequence,
            prerequisites={"altitude": 3000, "energy_state": "HIGH"},
            effectiveness_conditions={"emergency_escape": True, "overshoot_defense": True},
            energy_cost=0.8,
            risk_level=0.6
        )
    
    def _initialize_offensive_maneuvers(self):
        """Initialize offensive BFM maneuvers"""
        
        # Immelmann Turn - Offensive repositioning
        immelmann_sequence = []
        for step in range(25):
            if step < 15:  # Climb and pull
                pitch = 0.7
                roll = 0.0
                yaw = 0.0
            else:  # Roll out at top
                pitch = 0.0
                roll = 0.8
                yaw = 0.0
            fire = 0.0
            
            immelmann_sequence.append(np.array([pitch, roll, yaw, fire]))
        
        self.macro_actions[MacroActionType.IMMELMANN_TURN] = MacroActionSequence(
            name="Immelmann Turn",
            description="Climbing turn to reverse direction and gain altitude",
            duration_steps=25,
            control_sequence=immelmann_sequence,
            prerequisites={"energy_state": "HIGH", "altitude": 2000},
            effectiveness_conditions={"need_altitude": True, "reverse_direction": True},
            energy_cost=0.6,
            risk_level=0.3
        )
        
        # High Yo-Yo - Offensive BFM
        high_yo_yo_sequence = []
        for step in range(20):
            # Climb, turn, dive sequence
            if step < 8:  # Climb phase
                pitch = 0.5
                roll = 0.3
                yaw = 0.2
            elif step < 15:  # Turn phase
                pitch = 0.0
                roll = 0.6
                yaw = 0.4
            else:  # Dive phase
                pitch = -0.4
                roll = 0.0
                yaw = 0.0
            fire = 0.0
            
            high_yo_yo_sequence.append(np.array([pitch, roll, yaw, fire]))
        
        self.macro_actions[MacroActionType.HIGH_YO_YO] = MacroActionSequence(
            name="High Yo-Yo",
            description="Climbing turn to gain energy and position advantage",
            duration_steps=20,
            control_sequence=high_yo_yo_sequence,
            prerequisites={"energy_state": "MEDIUM", "lag_pursuit": True},
            effectiveness_conditions={"need_position_advantage": True, "energy_fight": True},
            energy_cost=0.4,
            risk_level=0.2
        )
    
    def _initialize_energy_maneuvers(self):
        """Initialize energy management maneuvers"""
        
        # Zoom Climb - Convert speed to altitude
        zoom_climb_sequence = []
        for step in range(15):
            # Aggressive climb to trade speed for altitude
            pitch = 0.8 - (step * 0.02)  # Decreasing pitch as speed bleeds
            roll = 0.0
            yaw = 0.0
            fire = 0.0
            
            zoom_climb_sequence.append(np.array([pitch, roll, yaw, fire]))
        
        self.macro_actions[MacroActionType.ZOOM_CLIMB] = MacroActionSequence(
            name="Zoom Climb",
            description="Convert airspeed to altitude for energy advantage",
            duration_steps=15,
            control_sequence=zoom_climb_sequence,
            prerequisites={"airspeed": "high", "altitude": "low"},
            effectiveness_conditions={"need_altitude": True, "excess_speed": True},
            energy_cost=-0.2,  # Actually gains potential energy
            risk_level=0.2
        )
        
        # Energy Dive - Convert altitude to speed
        energy_dive_sequence = []
        for step in range(12):
            # Controlled dive to gain speed
            pitch = -0.6 + (step * 0.03)  # Decreasing dive angle
            roll = 0.0
            yaw = 0.0
            fire = 0.0
            
            energy_dive_sequence.append(np.array([pitch, roll, yaw, fire]))
        
        self.macro_actions[MacroActionType.ENERGY_DIVE] = MacroActionSequence(
            name="Energy Dive", 
            description="Convert altitude to airspeed for energy advantage",
            duration_steps=12,
            control_sequence=energy_dive_sequence,
            prerequisites={"altitude": "high", "airspeed": "low"},
            effectiveness_conditions={"need_speed": True, "excess_altitude": True},
            energy_cost=-0.3,  # Gains kinetic energy
            risk_level=0.3
        )
    
    def _initialize_tactical_maneuvers(self):
        """Initialize tactical positioning maneuvers"""
        
        # Crank Maneuver - Maintain missile range while maneuvering
        crank_sequence = []
        for step in range(18):
            # Gentle turn to maintain radar lock while maneuvering
            pitch = 0.1
            roll = 0.4
            yaw = 0.2
            fire = 1.0 if step == 10 else 0.0  # Fire in middle of maneuver
            
            crank_sequence.append(np.array([pitch, roll, yaw, fire]))
        
        self.macro_actions[MacroActionType.CRANK_MANEUVER] = MacroActionSequence(
            name="Crank Maneuver",
            description="Maintain missile range while maneuvering for shot",
            duration_steps=18,
            control_sequence=crank_sequence,
            prerequisites={"locked": True, "bvr_range": True},
            effectiveness_conditions={"missile_shot_opportunity": True},
            energy_cost=0.3,
            risk_level=0.2
        )
        
        # Notch Maneuver - Defeat incoming missile
        notch_sequence = []
        for step in range(12):
            # 90-degree turn to notch incoming missile
            pitch = 0.0
            roll = 0.7
            yaw = 0.5
            fire = 0.0
            
            notch_sequence.append(np.array([pitch, roll, yaw, fire]))
        
        self.macro_actions[MacroActionType.NOTCH_MANEUVER] = MacroActionSequence(
            name="Notch Maneuver",
            description="90-degree turn to defeat incoming radar missile",
            duration_steps=12,
            control_sequence=notch_sequence,
            prerequisites={"missile_incoming": True},
            effectiveness_conditions={"radar_missile_threat": True},
            energy_cost=0.5,
            risk_level=0.4
        )
    
    def _print_available_maneuvers(self):
        """Print available macro actions"""
        print(f"\n{'='*60}")
        print("AVAILABLE MACRO ACTIONS")
        print(f"{'='*60}")
        
        categories = {
            'Defensive': [MacroActionType.DEFENSIVE_SPIRAL, MacroActionType.BARREL_ROLL, 
                         MacroActionType.SPLIT_S, MacroActionType.NOTCH_MANEUVER],
            'Offensive': [MacroActionType.IMMELMANN_TURN, MacroActionType.HIGH_YO_YO, 
                         MacroActionType.LOW_YO_YO, MacroActionType.LEAD_PURSUIT],
            'Energy': [MacroActionType.ZOOM_CLIMB, MacroActionType.ENERGY_DIVE, 
                      MacroActionType.LEVEL_ACCELERATION],
            'Tactical': [MacroActionType.CRANK_MANEUVER, MacroActionType.BEAM_MANEUVER, 
                        MacroActionType.DRAG_MANEUVER]
        }
        
        for category, actions in categories.items():
            print(f"\n{category} Maneuvers:")
            for action_type in actions:
                if action_type in self.macro_actions:
                    action = self.macro_actions[action_type]
                    print(f"  • {action.name}: {action.description}")
    
    def get_macro_action(self, action_type: MacroActionType) -> Optional[MacroActionSequence]:
        """Get macro action sequence"""
        return self.macro_actions.get(action_type)
    
    def get_available_actions(self, current_state: Dict[str, Any]) -> List[MacroActionType]:
        """Get macro actions available in current state"""
        
        available = []
        
        for action_type, action_seq in self.macro_actions.items():
            if self._check_prerequisites(action_seq.prerequisites, current_state):
                available.append(action_type)
        
        return available
    
    def _check_prerequisites(self, prerequisites: Dict[str, Any], 
                           current_state: Dict[str, Any]) -> bool:
        """Check if prerequisites are met for macro action"""
        
        for req_key, req_value in prerequisites.items():
            state_value = current_state.get(req_key)
            
            if req_key == "threat_level":
                if state_value is None or state_value < req_value:
                    return False
            elif req_key == "energy_state":
                if state_value != req_value:
                    return False
            elif req_key == "distance":
                if state_value is None or state_value > req_value:
                    return False
            # Add more prerequisite checks as needed
        
        return True


class HierarchicalActionSpace:
    """
    Hierarchical action space combining low-level control with high-level tactical maneuvers.
    
    Action Space Structure:
    - Low-level: Continuous control [pitch, roll, yaw, fire] ∈ [-1,1]⁴
    - High-level: Discrete macro actions (defensive spiral, barrel roll, etc.)
    - Meta-level: Action selection strategy (when to use macro vs micro actions)
    """
    
    def __init__(self, enable_macro_actions: bool = True, macro_action_probability: float = 0.1):
        """
        Initialize hierarchical action space
        
        Args:
            enable_macro_actions: Enable high-level macro actions
            macro_action_probability: Probability of selecting macro action vs micro action
        """
        self.enable_macro_actions = enable_macro_actions
        self.macro_action_probability = macro_action_probability
        
        # Low-level continuous action space
        self.micro_action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # High-level macro action library
        self.macro_library = MacroActionLibrary()
        
        # Current macro action state
        self.current_macro_action = None
        self.macro_action_step = 0
        self.macro_action_remaining = 0
        
        # Action selection strategy
        self.action_selection_history = deque(maxlen=100)
        self.macro_action_effectiveness = {}
        
        # Combined action space
        if enable_macro_actions:
            # Discrete choice: 0=micro, 1-N=macro actions
            num_macro_actions = len(self.macro_library.macro_actions)
            self.action_selection_space = gym.spaces.Discrete(num_macro_actions + 1)
            
            print(f"[HIERARCHICAL] Initialized with {num_macro_actions} macro actions")
        else:
            self.action_selection_space = None
            print(f"[HIERARCHICAL] Micro actions only")
        
        # Performance tracking
        self.performance_metrics = {
            'micro_actions_taken': 0,
            'macro_actions_taken': 0,
            'macro_action_success_rate': 0.0,
            'action_effectiveness': {}
        }
    
    def select_action(self, agent_action: np.ndarray, current_state: Dict[str, Any],
                     tactical_context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Select action using hierarchical strategy
        
        Args:
            agent_action: Raw agent action (continuous)
            current_state: Current tactical state
            tactical_context: Additional tactical context
        
        Returns:
            (final_action, action_info)
        """
        
        action_info = {
            'action_type': 'micro',
            'macro_action': None,
            'remaining_steps': 0,
            'effectiveness_prediction': 0.0
        }
        
        # If currently executing macro action, continue it
        if self.macro_action_remaining > 0:
            final_action = self._continue_macro_action()
            action_info.update({
                'action_type': 'macro_continuation',
                'macro_action': self.current_macro_action.name,
                'remaining_steps': self.macro_action_remaining
            })
            return final_action, action_info
        
        # Decide whether to use macro action
        if (self.enable_macro_actions and 
            self._should_use_macro_action(current_state, tactical_context)):
            
            # Select appropriate macro action
            macro_action_type = self._select_macro_action(current_state, tactical_context)
            
            if macro_action_type:
                final_action = self._start_macro_action(macro_action_type)
                action_info.update({
                    'action_type': 'macro_start',
                    'macro_action': self.current_macro_action.name,
                    'remaining_steps': self.macro_action_remaining
                })
                
                self.performance_metrics['macro_actions_taken'] += 1
                return final_action, action_info
        
        # Use micro action (default)
        final_action = self._process_micro_action(agent_action, current_state)
        action_info['effectiveness_prediction'] = self._predict_action_effectiveness(
            final_action, current_state
        )
        
        self.performance_metrics['micro_actions_taken'] += 1
        return final_action, action_info
    
    def _should_use_macro_action(self, current_state: Dict[str, Any],
                               tactical_context: Dict[str, Any] = None) -> bool:
        """Decide whether to use macro action based on tactical situation"""
        
        # High-level decision logic
        threat_level = current_state.get('threat_level', 0)
        engagement_phase = current_state.get('engagement_phase', 'UNKNOWN')
        
        # Emergency situations - high probability of macro action
        if threat_level > 0.8:
            return np.random.random() < 0.7  # 70% chance in emergency
        
        # Specific tactical situations
        if engagement_phase == 'WVR' and current_state.get('energy_state') == 'HIGH':
            return np.random.random() < 0.4  # 40% chance for offensive maneuvers
        
        if current_state.get('missile_incoming', False):
            return np.random.random() < 0.8  # 80% chance for defensive maneuvers
        
        # Base probability
        return np.random.random() < self.macro_action_probability
    
    def _select_macro_action(self, current_state: Dict[str, Any],
                           tactical_context: Dict[str, Any] = None) -> Optional[MacroActionType]:
        """Select appropriate macro action for current situation"""
        
        available_actions = self.macro_library.get_available_actions(current_state)
        
        if not available_actions:
            return None
        
        # Situation-based selection
        threat_level = current_state.get('threat_level', 0)
        engagement_phase = current_state.get('engagement_phase', 'UNKNOWN')
        
        # Emergency defensive maneuvers
        if threat_level > 0.8 or current_state.get('missile_incoming', False):
            defensive_actions = [a for a in available_actions 
                               if a in [MacroActionType.DEFENSIVE_SPIRAL, 
                                       MacroActionType.NOTCH_MANEUVER,
                                       MacroActionType.SPLIT_S]]
            if defensive_actions:
                return np.random.choice(defensive_actions)
        
        # WVR offensive maneuvers
        if engagement_phase == 'WVR' and current_state.get('energy_state') == 'HIGH':
            offensive_actions = [a for a in available_actions
                               if a in [MacroActionType.HIGH_YO_YO,
                                       MacroActionType.BARREL_ROLL,
                                       MacroActionType.IMMELMANN_TURN]]
            if offensive_actions:
                return np.random.choice(offensive_actions)
        
        # BVR tactical maneuvers
        if engagement_phase == 'BVR' and current_state.get('locked', False):
            tactical_actions = [a for a in available_actions
                              if a in [MacroActionType.CRANK_MANEUVER]]
            if tactical_actions:
                return np.random.choice(tactical_actions)
        
        # Default: random selection from available
        return np.random.choice(available_actions) if available_actions else None
    
    def _start_macro_action(self, action_type: MacroActionType) -> np.ndarray:
        """Start executing a macro action"""
        
        self.current_macro_action = self.macro_library.get_macro_action(action_type)
        self.macro_action_step = 0
        self.macro_action_remaining = self.current_macro_action.duration_steps
        
        print(f"[MACRO] Starting {self.current_macro_action.name} "
              f"({self.current_macro_action.duration_steps} steps)")
        
        # Return first action in sequence
        return self._continue_macro_action()
    
    def _continue_macro_action(self) -> np.ndarray:
        """Continue executing current macro action"""
        
        if (self.current_macro_action is None or 
            self.macro_action_step >= len(self.current_macro_action.control_sequence)):
            # Macro action complete
            self.macro_action_remaining = 0
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Get next action in sequence
        action = self.current_macro_action.control_sequence[self.macro_action_step]
        self.macro_action_step += 1
        self.macro_action_remaining -= 1
        
        # Check if macro action is complete
        if self.macro_action_remaining <= 0:
            print(f"[MACRO] Completed {self.current_macro_action.name}")
            self.current_macro_action = None
            self.macro_action_step = 0
        
        return action
    
    def _process_micro_action(self, agent_action: np.ndarray, 
                            current_state: Dict[str, Any]) -> np.ndarray:
        """Process and potentially modify micro action"""
        
        # Apply safety limits and tactical modifications
        processed_action = agent_action.copy()
        
        # Safety limits based on current state
        threat_level = current_state.get('threat_level', 0)
        energy_state = current_state.get('energy_state', 'MEDIUM')
        
        # Limit aggressive maneuvers in low energy state
        if energy_state == 'LOW':
            processed_action[:3] *= 0.7  # Reduce control authority
        
        # Encourage defensive maneuvers under high threat
        if threat_level > 0.7:
            # Bias toward defensive actions
            processed_action[0] = min(processed_action[0], 0.0)  # Limit climb
            processed_action[1] = np.sign(processed_action[1]) * min(abs(processed_action[1]), 0.8)
        
        # Firing logic enhancement
        locked = current_state.get('locked', False)
        distance = current_state.get('distance', float('inf'))
        
        if processed_action[3] > 0.5:  # Fire action
            if not locked or distance > 15000 or distance < 1000:
                processed_action[3] = 0.0  # Prevent bad shots
        
        return processed_action
    
    def _predict_action_effectiveness(self, action: np.ndarray, 
                                    current_state: Dict[str, Any]) -> float:
        """Predict effectiveness of action in current state"""
        
        effectiveness = 0.5  # Base effectiveness
        
        # Distance-based effectiveness
        distance = current_state.get('distance', 0)
        engagement_phase = current_state.get('engagement_phase', 'UNKNOWN')
        
        if engagement_phase == 'BVR' and 8000 <= distance <= 15000:
            effectiveness += 0.2  # Good for BVR
        elif engagement_phase == 'WVR' and 2000 <= distance <= 5000:
            effectiveness += 0.2  # Good for WVR
        
        # Action appropriateness
        if action[3] > 0.5:  # Fire action
            if current_state.get('locked', False) and 2000 <= distance <= 12000:
                effectiveness += 0.3  # Good shot opportunity
            else:
                effectiveness -= 0.2  # Poor shot opportunity
        
        # Control input quality
        control_magnitude = np.linalg.norm(action[:3])
        if 0.2 <= control_magnitude <= 0.6:
            effectiveness += 0.1  # Appropriate control inputs
        elif control_magnitude > 0.8:
            effectiveness -= 0.1  # Excessive control inputs
        
        return max(0.0, min(1.0, effectiveness))
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Get information about the hierarchical action space"""
        
        return {
            'micro_action_space': {
                'type': 'continuous',
                'shape': self.micro_action_space.shape,
                'low': self.micro_action_space.low.tolist(),
                'high': self.micro_action_space.high.tolist()
            },
            'macro_actions_enabled': self.enable_macro_actions,
            'available_macro_actions': len(self.macro_library.macro_actions),
            'macro_action_types': [action_type.value for action_type in self.macro_library.macro_actions.keys()],
            'current_macro_action': self.current_macro_action.name if self.current_macro_action else None,
            'macro_steps_remaining': self.macro_action_remaining
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get action space performance metrics"""
        
        total_actions = (self.performance_metrics['micro_actions_taken'] + 
                        self.performance_metrics['macro_actions_taken'])
        
        if total_actions > 0:
            macro_usage_rate = self.performance_metrics['macro_actions_taken'] / total_actions
        else:
            macro_usage_rate = 0.0
        
        return {
            **self.performance_metrics,
            'total_actions': total_actions,
            'macro_usage_rate': macro_usage_rate,
            'average_effectiveness': np.mean(list(self.performance_metrics['action_effectiveness'].values())) if self.performance_metrics['action_effectiveness'] else 0.0
        }


class EnhancedActionProcessor:
    """
    Enhanced action processor that integrates hierarchical actions with RL training
    """
    
    def __init__(self, hierarchical_action_space: HierarchicalActionSpace,
                 llm_assistant=None):
        """
        Initialize enhanced action processor
        
        Args:
            hierarchical_action_space: Hierarchical action space
            llm_assistant: Optional LLM assistant for action guidance
        """
        self.action_space = hierarchical_action_space
        self.llm_assistant = llm_assistant
        
        # Action history for analysis
        self.action_history = deque(maxlen=1000)
        self.effectiveness_history = deque(maxlen=1000)
        
        print(f"[ACTION PROCESSOR] Enhanced processor initialized")
    
    def process_agent_action(self, raw_action: np.ndarray, 
                           current_state: Dict[str, Any],
                           tactical_features: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process raw agent action through hierarchical action space
        
        Args:
            raw_action: Raw action from RL agent
            current_state: Current environment state
            tactical_features: Additional tactical features
        
        Returns:
            (processed_action, action_metadata)
        """
        
        # Get LLM tactical guidance if available
        llm_guidance = None
        if self.llm_assistant and tactical_features:
            try:
                shaping_delta, llm_response = self.llm_assistant.request_shaping(tactical_features)
                llm_guidance = llm_response
            except Exception as e:
                print(f"[ACTION PROCESSOR] LLM guidance failed: {e}")
        
        # Process through hierarchical action space
        final_action, action_info = self.action_space.select_action(
            raw_action, current_state, tactical_features
        )
        
        # Enhance action info with LLM guidance
        if llm_guidance:
            action_info['llm_guidance'] = llm_guidance
            action_info['llm_shaping'] = llm_guidance.get('shaping_delta', 0.0)
        
        # Record action for analysis
        self.action_history.append({
            'raw_action': raw_action.copy(),
            'final_action': final_action.copy(),
            'action_info': action_info.copy(),
            'state': current_state.copy(),
            'timestamp': time.time()
        })
        
        return final_action, action_info
    
    def update_action_effectiveness(self, action_metadata: Dict[str, Any], 
                                  reward: float, success: bool):
        """Update action effectiveness based on outcomes"""
        
        action_type = action_metadata.get('action_type', 'micro')
        
        # Record effectiveness
        self.effectiveness_history.append({
            'action_type': action_type,
            'macro_action': action_metadata.get('macro_action'),
            'reward': reward,
            'success': success,
            'timestamp': time.time()
        })
        
        # Update macro action effectiveness tracking
        if action_type.startswith('macro'):
            macro_name = action_metadata.get('macro_action', 'unknown')
            if macro_name not in self.action_space.macro_action_effectiveness:
                self.action_space.macro_action_effectiveness[macro_name] = []
            
            self.action_space.macro_action_effectiveness[macro_name].append({
                'reward': reward,
                'success': success
            })
    
    def get_action_analysis(self) -> Dict[str, Any]:
        """Get comprehensive action analysis"""
        
        if not self.action_history:
            return {'status': 'no_data'}
        
        # Analyze action types
        recent_actions = list(self.action_history)[-100:]  # Last 100 actions
        action_types = [a['action_info']['action_type'] for a in recent_actions]
        
        micro_count = sum(1 for at in action_types if at == 'micro')
        macro_count = sum(1 for at in action_types if at.startswith('macro'))
        
        # Analyze effectiveness
        if self.effectiveness_history:
            recent_effectiveness = list(self.effectiveness_history)[-50:]
            avg_reward = np.mean([e['reward'] for e in recent_effectiveness])
            success_rate = np.mean([e['success'] for e in recent_effectiveness])
        else:
            avg_reward = 0.0
            success_rate = 0.0
        
        return {
            'total_actions_analyzed': len(self.action_history),
            'recent_micro_actions': micro_count,
            'recent_macro_actions': macro_count,
            'macro_usage_rate': macro_count / (micro_count + macro_count) if (micro_count + macro_count) > 0 else 0,
            'average_reward': avg_reward,
            'success_rate': success_rate,
            'action_space_info': self.action_space.get_action_space_info(),
            'performance_metrics': self.action_space.get_performance_metrics()
        }


def create_optimized_training_environment(base_env_class, enable_all_optimizations: bool = True):
    """
    Create performance-optimized training environment wrapper
    
    Args:
        base_env_class: Base environment class
        enable_all_optimizations: Enable all performance optimizations
    
    Returns:
        Optimized environment class
    """
    
    class OptimizedTrainingEnvironment(base_env_class):
        """Environment wrapper with performance optimizations"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Add performance optimizations
            if enable_all_optimizations:
                self.hierarchical_actions = HierarchicalActionSpace(
                    enable_macro_actions=True,
                    macro_action_probability=0.15
                )
                
                self.action_processor = EnhancedActionProcessor(
                    self.hierarchical_actions
                )
                
                print("[OPTIMIZED ENV] All performance optimizations enabled")
            else:
                self.hierarchical_actions = None
                self.action_processor = None
        
        def step(self, action):
            """Optimized step with hierarchical actions"""
            
            if self.action_processor:
                # Process action through hierarchical system
                current_state = self._get_current_state_dict()
                processed_action, action_info = self.action_processor.process_agent_action(
                    action, current_state
                )
                
                # Step with processed action
                result = super().step(processed_action)
                obs, reward, terminated, truncated, info = result
                
                # Update action effectiveness
                success = info.get('success', reward > 0)
                self.action_processor.update_action_effectiveness(action_info, reward, success)
                
                # Add action info to environment info
                info['action_info'] = action_info
                
                return obs, reward, terminated, truncated, info
            else:
                # Standard step
                return super().step(action)
        
        def _get_current_state_dict(self) -> Dict[str, Any]:
            """Get current state as dictionary for action processing"""
            
            # Extract state information from environment
            # This would be customized based on specific environment
            return {
                'distance': getattr(self, 'distance', 8000),
                'threat_level': getattr(self, 'threat_level', 0.3),
                'engagement_phase': getattr(self, 'engagement_phase', 'BVR'),
                'energy_state': getattr(self, 'energy_state', 'MEDIUM'),
                'locked': getattr(self, 'locked', False)
            }
        
        def get_optimization_summary(self) -> Dict[str, Any]:
            """Get optimization performance summary"""
            
            summary = {
                'optimizations_enabled': enable_all_optimizations,
                'hierarchical_actions': self.hierarchical_actions is not None
            }
            
            if self.action_processor:
                summary['action_analysis'] = self.action_processor.get_action_analysis()
            
            return summary
    
    return OptimizedTrainingEnvironment


if __name__ == "__main__":
    print("Real-time Performance Optimization for Harfang RL-LLM")
    print("Provides async LLM calls, situation caching, and hierarchical actions")
    
    # Example usage
    print("\nCreating hierarchical action space...")
    action_space = HierarchicalActionSpace(enable_macro_actions=True)
    
    print(f"Action space info: {action_space.get_action_space_info()}")
