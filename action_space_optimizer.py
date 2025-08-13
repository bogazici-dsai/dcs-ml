# action_space_optimizer.py - Dynamic action space optimization based on LLM feedback
import json
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque


class ActionSpaceOptimizer:
    """
    Optimizes action space based on LLM feedback and performance data.
    
    This system analyzes LLM recommendations and tactical performance to suggest
    modifications to the action space, including new high-level actions and
    parameter adjustments.
    """
    
    def __init__(self, initial_action_bounds: Dict[str, Tuple[float, float]], 
                 optimization_window: int = 1000,
                 min_samples_for_optimization: int = 100):
        """
        Initialize action space optimizer
        
        Args:
            initial_action_bounds: Initial action space bounds {'action_name': (min, max)}
            optimization_window: Number of recent steps to consider for optimization
            min_samples_for_optimization: Minimum samples needed before suggesting changes
        """
        self.initial_bounds = initial_action_bounds.copy()
        self.current_bounds = initial_action_bounds.copy()
        self.optimization_window = optimization_window
        self.min_samples = min_samples_for_optimization
        
        # Data collection for optimization
        self.action_effectiveness = defaultdict(lambda: deque(maxlen=optimization_window))
        self.llm_recommendations = deque(maxlen=optimization_window)
        self.performance_metrics = deque(maxlen=optimization_window)
        self.action_usage_frequency = defaultdict(lambda: deque(maxlen=optimization_window))
        
        # High-level action proposals
        self.proposed_macro_actions = {}
        self.macro_action_performance = defaultdict(list)
        
        # Optimization history
        self.optimization_history = []
        self.last_optimization_time = time.time()
        
        print("[ACTION OPTIMIZER] Initialized with action bounds:", self.current_bounds)

    def record_step(self, action: np.ndarray, reward: float, llm_response: Dict[str, Any],
                   tactical_metrics: Dict[str, Any], step_info: Dict[str, Any]):
        """
        Record a step for action space optimization analysis
        
        Args:
            action: The action taken [pitch, roll, yaw, fire]
            reward: The reward received (including LLM shaping)
            llm_response: LLM response with recommendations
            tactical_metrics: Tactical situation metrics
            step_info: Additional step information
        """
        
        # Record action effectiveness
        action_vector = action.tolist() if hasattr(action, 'tolist') else list(action)
        self.action_effectiveness['pitch'].append((action_vector[0], reward))
        self.action_effectiveness['roll'].append((action_vector[1], reward))
        self.action_effectiveness['yaw'].append((action_vector[2], reward))
        self.action_effectiveness['fire'].append((action_vector[3], reward))
        
        # Record action usage frequency
        self.action_usage_frequency['pitch'].append(abs(action_vector[0]))
        self.action_usage_frequency['roll'].append(abs(action_vector[1]))
        self.action_usage_frequency['yaw'].append(abs(action_vector[2]))
        self.action_usage_frequency['fire'].append(action_vector[3] > 0)
        
        # Store LLM recommendations
        self.llm_recommendations.append({
            'timestamp': time.time(),
            'critique': llm_response.get('critique', ''),
            'recommendations': llm_response.get('recommendations', {}),
            'action_space_feedback': llm_response.get('action_space_feedback', {}),
            'shaping_delta': llm_response.get('shaping_delta', 0)
        })
        
        # Store performance metrics
        self.performance_metrics.append({
            'timestamp': time.time(),
            'reward': reward,
            'tactical_situation': tactical_metrics.get('tactical_situation', 'UNKNOWN'),
            'engagement_phase': tactical_metrics.get('engagement_phase', 'UNKNOWN'),
            'threat_level': tactical_metrics.get('threat_level', 0),
            'in_firing_envelope': tactical_metrics.get('in_firing_envelope', False),
            'energy_state': tactical_metrics.get('energy_state', 'UNKNOWN'),
            'distance': tactical_metrics.get('distance', 0),
            'locked': tactical_metrics.get('locked', 0) > 0
        })
        
        # Process any macro action recommendations
        self._process_macro_action_recommendations(llm_response)

    def _process_macro_action_recommendations(self, llm_response: Dict[str, Any]):
        """Process LLM recommendations for new macro actions"""
        
        action_space_ops = llm_response.get('action_space_ops', {})
        if not action_space_ops:
            return
        
        # Process add recommendations
        add_actions = action_space_ops.get('add', [])
        for action_name in add_actions:
            if action_name not in self.proposed_macro_actions:
                self.proposed_macro_actions[action_name] = {
                    'proposed_count': 1,
                    'first_proposed': time.time(),
                    'contexts': [llm_response.get('critique', '')]
                }
            else:
                self.proposed_macro_actions[action_name]['proposed_count'] += 1
                self.proposed_macro_actions[action_name]['contexts'].append(
                    llm_response.get('critique', ''))

    def analyze_action_effectiveness(self) -> Dict[str, Any]:
        """Analyze action effectiveness and suggest optimizations"""
        
        if len(self.performance_metrics) < self.min_samples:
            return {'status': 'insufficient_data', 'samples': len(self.performance_metrics)}
        
        analysis = {
            'timestamp': time.time(),
            'samples_analyzed': len(self.performance_metrics),
            'action_analysis': {},
            'macro_action_proposals': {},
            'optimization_suggestions': []
        }
        
        # Analyze each action dimension
        for action_name in ['pitch', 'roll', 'yaw', 'fire']:
            action_analysis = self._analyze_action_dimension(action_name)
            analysis['action_analysis'][action_name] = action_analysis
        
        # Analyze macro action proposals
        analysis['macro_action_proposals'] = self._analyze_macro_proposals()
        
        # Generate optimization suggestions
        analysis['optimization_suggestions'] = self._generate_optimization_suggestions(analysis)
        
        return analysis

    def _analyze_action_dimension(self, action_name: str) -> Dict[str, Any]:
        """Analyze effectiveness of a specific action dimension"""
        
        if action_name not in self.action_effectiveness:
            return {'status': 'no_data'}
        
        action_data = list(self.action_effectiveness[action_name])
        if len(action_data) < self.min_samples:
            return {'status': 'insufficient_data'}
        
        # Extract actions and rewards
        actions = [data[0] for data in action_data]
        rewards = [data[1] for data in action_data]
        
        # Basic statistics
        action_mean = np.mean(actions)
        action_std = np.std(actions)
        reward_mean = np.mean(rewards)
        
        # Effectiveness analysis
        # Divide actions into high/low magnitude and compare rewards
        action_threshold = np.median(np.abs(actions))
        high_magnitude_indices = [i for i, a in enumerate(actions) if abs(a) > action_threshold]
        low_magnitude_indices = [i for i, a in enumerate(actions) if abs(a) <= action_threshold]
        
        high_mag_rewards = [rewards[i] for i in high_magnitude_indices] if high_magnitude_indices else [0]
        low_mag_rewards = [rewards[i] for i in low_magnitude_indices] if low_magnitude_indices else [0]
        
        effectiveness_ratio = np.mean(high_mag_rewards) / max(np.mean(low_mag_rewards), 0.001)
        
        # Usage frequency analysis
        usage_freq = list(self.action_usage_frequency[action_name])
        usage_mean = np.mean(usage_freq) if usage_freq else 0
        
        return {
            'status': 'analyzed',
            'mean_value': action_mean,
            'std_value': action_std,
            'mean_reward': reward_mean,
            'effectiveness_ratio': effectiveness_ratio,
            'usage_frequency': usage_mean,
            'samples': len(action_data),
            'current_bounds': self.current_bounds.get(action_name, (-1, 1)),
            'recommendation': self._get_action_recommendation(
                action_name, effectiveness_ratio, usage_mean, action_std
            )
        }

    def _get_action_recommendation(self, action_name: str, effectiveness_ratio: float,
                                 usage_frequency: float, action_std: float) -> Dict[str, Any]:
        """Generate recommendation for action dimension adjustment"""
        
        recommendations = []
        
        # High effectiveness but low usage - consider increasing sensitivity
        if effectiveness_ratio > 1.2 and usage_frequency < 0.3:
            recommendations.append({
                'type': 'increase_sensitivity',
                'reason': 'High effectiveness but low usage',
                'suggested_multiplier': 1.2
            })
        
        # Low effectiveness but high usage - consider decreasing sensitivity
        elif effectiveness_ratio < 0.8 and usage_frequency > 0.7:
            recommendations.append({
                'type': 'decrease_sensitivity',
                'reason': 'Low effectiveness but high usage',
                'suggested_multiplier': 0.8
            })
        
        # Very low standard deviation - actions too constrained
        if action_std < 0.1:
            recommendations.append({
                'type': 'increase_exploration',
                'reason': 'Actions too constrained, low variation',
                'suggested_action': 'Add exploration noise or expand bounds'
            })
        
        # Special recommendations for fire action
        if action_name == 'fire':
            fire_usage = usage_frequency
            if fire_usage < 0.1:
                recommendations.append({
                    'type': 'encourage_firing',
                    'reason': 'Very low missile usage',
                    'suggested_action': 'Adjust firing conditions or rewards'
                })
            elif fire_usage > 0.5:
                recommendations.append({
                    'type': 'discourage_wasteful_firing',
                    'reason': 'High missile usage, potentially wasteful',
                    'suggested_action': 'Strengthen lock requirements'
                })
        
        return {
            'recommendations': recommendations,
            'priority': 'high' if len(recommendations) > 1 else 'medium' if recommendations else 'low'
        }

    def _analyze_macro_proposals(self) -> Dict[str, Any]:
        """Analyze proposed macro actions from LLM"""
        
        proposals = {}
        
        for action_name, data in self.proposed_macro_actions.items():
            if data['proposed_count'] >= 3:  # Threshold for serious consideration
                proposals[action_name] = {
                    'proposal_count': data['proposed_count'],
                    'contexts': data['contexts'][-3:],  # Last 3 contexts
                    'feasibility': self._assess_macro_action_feasibility(action_name),
                    'recommendation': 'implement' if data['proposed_count'] >= 5 else 'monitor'
                }
        
        return proposals

    def _assess_macro_action_feasibility(self, action_name: str) -> str:
        """Assess feasibility of implementing a macro action for air combat"""
        
        # Air combat tactical maneuvers and their implementation difficulty
        high_feasibility_actions = [
            'defensive_spiral', 'barrel_roll', 'split_s', 'immelmann',
            'energy_climb', 'defensive_turn', 'notch_maneuver', 'beam_maneuver',
            'break_turn', 'slice_back', 'vertical_scissors'
        ]
        
        medium_feasibility_actions = [
            'bvr_crank', 'merge_geometry', 'offensive_bfm', 'defensive_bfm',
            'gun_tracking', 'missile_defense', 'chaff_flare', 'notch_and_crank',
            'lead_pursuit', 'lag_pursuit', 'pure_pursuit'
        ]
        
        complex_actions = [
            'thach_weave', 'loose_deuce', 'wall_formation', 'bracket_attack',
            'pincer_movement', 'energy_sustaining_turn'
        ]
        
        action_lower = action_name.lower()
        
        if any(tactical in action_lower for tactical in high_feasibility_actions):
            return 'high'
        elif any(tactical in action_lower for tactical in medium_feasibility_actions):
            return 'medium'
        elif any(tactical in action_lower for tactical in complex_actions):
            return 'low'  # Complex multi-aircraft maneuvers
        elif any(keyword in action_lower for keyword in ['fire', 'missile', 'gun', 'weapon']):
            return 'medium'  # Weapons employment
        else:
            return 'low'

    def _generate_optimization_suggestions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive optimization suggestions"""
        
        suggestions = []
        
        # Action space boundary adjustments
        for action_name, action_analysis in analysis['action_analysis'].items():
            if action_analysis.get('status') == 'analyzed':
                recommendations = action_analysis['recommendation']['recommendations']
                for rec in recommendations:
                    if rec['type'] in ['increase_sensitivity', 'decrease_sensitivity']:
                        suggestions.append({
                            'type': 'boundary_adjustment',
                            'action': action_name,
                            'description': rec['reason'],
                            'implementation': f"Multiply {action_name} bounds by {rec['suggested_multiplier']}",
                            'priority': action_analysis['recommendation']['priority']
                        })
        
        # Macro action implementations
        for action_name, proposal in analysis['macro_action_proposals'].items():
            if proposal['recommendation'] == 'implement':
                suggestions.append({
                    'type': 'macro_action',
                    'action': action_name,
                    'description': f"Implement high-level action: {action_name}",
                    'implementation': f"Create composite action combining basic actions",
                    'priority': 'high' if proposal['feasibility'] == 'high' else 'medium'
                })
        
        # Performance-based suggestions
        recent_rewards = [m['reward'] for m in list(self.performance_metrics)[-50:]]
        if recent_rewards and np.mean(recent_rewards) < 0:
            suggestions.append({
                'type': 'reward_tuning',
                'action': 'overall',
                'description': 'Recent performance declining',
                'implementation': 'Review reward structure and LLM guidance parameters',
                'priority': 'high'
            })
        
        return suggestions

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        
        analysis = self.analyze_action_effectiveness()
        
        report = {
            'timestamp': time.time(),
            'session_summary': {
                'total_steps': len(self.performance_metrics),
                'optimization_window': self.optimization_window,
                'data_collection_period': time.time() - self.last_optimization_time
            },
            'action_space_analysis': analysis,
            'llm_feedback_summary': self._summarize_llm_feedback(),
            'performance_trends': self._analyze_performance_trends(),
            'implementation_priorities': self._prioritize_implementations(analysis),
            'next_steps': self._generate_next_steps(analysis)
        }
        
        return report

    def _summarize_llm_feedback(self) -> Dict[str, Any]:
        """Summarize LLM feedback patterns"""
        
        if not self.llm_recommendations:
            return {'status': 'no_data'}
        
        recent_recommendations = list(self.llm_recommendations)[-100:]  # Last 100
        
        # Count recommendation themes
        themes = defaultdict(int)
        avg_shaping = []
        
        for rec in recent_recommendations:
            critique = rec.get('critique', '').lower()
            shaping = rec.get('shaping_delta', 0)
            avg_shaping.append(shaping)
            
            # Simple keyword analysis
            if 'firing' in critique or 'missile' in critique:
                themes['weapons_employment'] += 1
            if 'energy' in critique or 'altitude' in critique:
                themes['energy_management'] += 1
            if 'positioning' in critique or 'aspect' in critique:
                themes['tactical_positioning'] += 1
            if 'defensive' in critique or 'threat' in critique:
                themes['defensive_maneuvers'] += 1
        
        return {
            'status': 'analyzed',
            'samples': len(recent_recommendations),
            'avg_shaping_delta': np.mean(avg_shaping) if avg_shaping else 0,
            'common_themes': dict(sorted(themes.items(), key=lambda x: x[1], reverse=True)),
            'positive_feedback_rate': len([s for s in avg_shaping if s > 0]) / max(len(avg_shaping), 1)
        }

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        
        if len(self.performance_metrics) < 50:
            return {'status': 'insufficient_data'}
        
        metrics = list(self.performance_metrics)
        recent_rewards = [m['reward'] for m in metrics[-50:]]
        early_rewards = [m['reward'] for m in metrics[:50]]
        
        return {
            'status': 'analyzed',
            'reward_trend': 'improving' if np.mean(recent_rewards) > np.mean(early_rewards) else 'declining',
            'recent_avg_reward': np.mean(recent_rewards),
            'early_avg_reward': np.mean(early_rewards),
            'improvement_rate': (np.mean(recent_rewards) - np.mean(early_rewards)) / max(abs(np.mean(early_rewards)), 0.1)
        }

    def _prioritize_implementations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize optimization implementations"""
        
        suggestions = analysis.get('optimization_suggestions', [])
        
        # Sort by priority and feasibility
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        
        prioritized = sorted(suggestions, 
                           key=lambda x: (priority_order.get(x.get('priority', 'low'), 1),
                                        x.get('type') == 'boundary_adjustment'),  # Prefer simple adjustments
                           reverse=True)
        
        return prioritized[:5]  # Top 5 priorities

    def _generate_next_steps(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable next steps"""
        
        next_steps = []
        
        # Check data sufficiency
        if analysis.get('samples_analyzed', 0) < self.min_samples * 2:
            next_steps.append("Continue data collection - need more samples for robust optimization")
        
        # Check for high-priority suggestions
        high_priority = [s for s in analysis.get('optimization_suggestions', []) 
                        if s.get('priority') == 'high']
        
        if high_priority:
            next_steps.append(f"Implement {len(high_priority)} high-priority optimizations")
        
        # Check macro action proposals
        macro_proposals = analysis.get('macro_action_proposals', {})
        ready_macros = [name for name, data in macro_proposals.items() 
                       if data.get('recommendation') == 'implement']
        
        if ready_macros:
            next_steps.append(f"Implement macro actions: {', '.join(ready_macros[:3])}")
        
        # Performance-based next steps
        performance = self._analyze_performance_trends()
        if performance.get('reward_trend') == 'declining':
            next_steps.append("Investigate performance decline - review recent changes")
        
        if not next_steps:
            next_steps.append("Continue monitoring - system performing within acceptable parameters")
        
        return next_steps

    def export_optimization_data(self, filepath: str):
        """Export optimization data for external analysis"""
        
        export_data = {
            'metadata': {
                'export_timestamp': time.time(),
                'optimization_window': self.optimization_window,
                'initial_bounds': self.initial_bounds,
                'current_bounds': self.current_bounds
            },
            'performance_data': list(self.performance_metrics),
            'llm_recommendations': list(self.llm_recommendations),
            'action_effectiveness': {k: list(v) for k, v in self.action_effectiveness.items()},
            'macro_proposals': dict(self.proposed_macro_actions),
            'optimization_history': self.optimization_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"[ACTION OPTIMIZER] Optimization data exported to {filepath}")
