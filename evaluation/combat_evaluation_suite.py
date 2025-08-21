# Comprehensive Combat Evaluation Suite for RL-LLM Performance Assessment
import numpy as np
import json
import time
import os
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class EvaluationScenario(Enum):
    """Standardized evaluation scenarios for combat assessment"""
    HEAD_ON_BVR = "head_on_bvr"
    BEAM_ASPECT_BVR = "beam_aspect_bvr"
    STERN_ASPECT_BVR = "stern_aspect_bvr"
    TAIL_CHASE_WVR = "tail_chase_wvr"
    MERGE_GEOMETRY = "merge_geometry"
    DEFENSIVE_SCENARIO = "defensive_scenario"
    MULTI_TARGET = "multi_target"
    LOW_ALTITUDE = "low_altitude"
    HIGH_ALTITUDE = "high_altitude"
    NOTCH_GEOMETRY = "notch_geometry"
    ENERGY_FIGHT = "energy_fight"
    GUNS_ONLY = "guns_only"


@dataclass
class ScenarioConfig:
    """Configuration for evaluation scenario"""
    name: str
    description: str
    initial_conditions: Dict[str, Any]
    success_criteria: Dict[str, float]
    difficulty_level: float  # 0.0 to 1.0
    expected_duration: int   # Expected steps to completion
    tactical_focus: List[str]  # Areas being evaluated


@dataclass
class EvaluationResult:
    """Results from scenario evaluation"""
    scenario_name: str
    success: bool
    score: float
    duration: int
    tactical_metrics: Dict[str, float]
    performance_breakdown: Dict[str, float]
    recommendations: List[str]
    detailed_log: Dict[str, Any]


class CombatEvaluationSuite:
    """
    Comprehensive evaluation suite for testing RL-LLM combat performance
    across standardized scenarios with consistent metrics and benchmarks.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize combat evaluation suite
        
        Args:
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        
        # Define evaluation scenarios
        self.scenarios = self._create_evaluation_scenarios()
        
        # Performance benchmarks
        self.benchmarks = self._create_performance_benchmarks()
        
        # Evaluation history
        self.evaluation_history = []
        self.agent_performance_profiles = {}
        
        print(f"[EVAL SUITE] Initialized with {len(self.scenarios)} evaluation scenarios")
        self._print_scenario_overview()
    
    def _create_evaluation_scenarios(self) -> Dict[EvaluationScenario, ScenarioConfig]:
        """Create standardized evaluation scenarios"""
        
        scenarios = {}
        
        # BVR Scenarios
        scenarios[EvaluationScenario.HEAD_ON_BVR] = ScenarioConfig(
            name="Head-On BVR Engagement",
            description="Classic head-on Beyond Visual Range missile engagement",
            initial_conditions={
                'ego_position': [0, 0, 8000],
                'enemy_position': [20000, 0, 8000],
                'ego_heading': 0,
                'enemy_heading': 180,
                'engagement_range': 20000,
                'mutual_detection': True
            },
            success_criteria={
                'enemy_destroyed': 1.0,
                'survival': 1.0,
                'missile_efficiency': 0.5,
                'time_limit': 300
            },
            difficulty_level=0.6,
            expected_duration=150,
            tactical_focus=['bvr_tactics', 'missile_employment', 'energy_management']
        )
        
        scenarios[EvaluationScenario.BEAM_ASPECT_BVR] = ScenarioConfig(
            name="Beam Aspect BVR",
            description="BVR engagement with 90-degree aspect angle",
            initial_conditions={
                'ego_position': [0, 0, 8000],
                'enemy_position': [15000, 15000, 8000],
                'ego_heading': 45,
                'enemy_heading': 225,
                'engagement_range': 21213,  # sqrt(15000² + 15000²)
                'aspect_angle': 90
            },
            success_criteria={
                'enemy_destroyed': 1.0,
                'survival': 1.0,
                'shot_accuracy': 0.4,
                'time_limit': 400
            },
            difficulty_level=0.7,
            expected_duration=200,
            tactical_focus=['geometry_management', 'lead_pursuit', 'missile_kinematics']
        )
        
        # WVR Scenarios  
        scenarios[EvaluationScenario.TAIL_CHASE_WVR] = ScenarioConfig(
            name="Tail Chase WVR",
            description="Within Visual Range tail chase engagement",
            initial_conditions={
                'ego_position': [0, 0, 5000],
                'enemy_position': [4000, 0, 5000],
                'ego_heading': 0,
                'enemy_heading': 0,
                'engagement_range': 4000,
                'chase_geometry': True
            },
            success_criteria={
                'enemy_destroyed': 1.0,
                'survival': 1.0,
                'guns_effectiveness': 0.3,
                'time_limit': 180
            },
            difficulty_level=0.8,
            expected_duration=120,
            tactical_focus=['wvr_maneuvering', 'guns_tracking', 'energy_fighting']
        )
        
        # Defensive Scenarios
        scenarios[EvaluationScenario.DEFENSIVE_SCENARIO] = ScenarioConfig(
            name="Defensive BFM",
            description="Defensive Basic Fighter Maneuvers under attack",
            initial_conditions={
                'ego_position': [0, 0, 6000],
                'enemy_position': [8000, 0, 7000],
                'ego_heading': 90,
                'enemy_heading': 270,
                'enemy_advantage': True,
                'missile_incoming': True
            },
            success_criteria={
                'survival': 1.0,
                'missile_defeat': 1.0,
                'counter_attack': 0.5,
                'time_limit': 240
            },
            difficulty_level=0.9,
            expected_duration=160,
            tactical_focus=['defensive_bfm', 'missile_defeat', 'survival_tactics']
        )
        
        # Multi-Target Scenarios
        scenarios[EvaluationScenario.MULTI_TARGET] = ScenarioConfig(
            name="Multi-Target Engagement",
            description="Engagement against multiple enemy aircraft",
            initial_conditions={
                'ego_position': [0, 0, 8000],
                'enemy_positions': [[12000, 0, 8000], [10000, 8000, 7000]],
                'threat_prioritization': True,
                'time_pressure': True
            },
            success_criteria={
                'primary_target_destroyed': 1.0,
                'survival': 1.0,
                'threat_prioritization': 0.8,
                'time_limit': 450
            },
            difficulty_level=1.0,
            expected_duration=300,
            tactical_focus=['multi_target', 'prioritization', 'situational_awareness']
        )
        
        return scenarios
    
    def _create_performance_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Create performance benchmarks for different agent types"""
        
        return {
            'random_agent': {
                'overall_success_rate': 0.05,
                'bvr_success_rate': 0.10,
                'wvr_success_rate': 0.02,
                'defensive_success_rate': 0.15,
                'average_score': 0.1
            },
            'rule_based_agent': {
                'overall_success_rate': 0.25,
                'bvr_success_rate': 0.35,
                'wvr_success_rate': 0.15,
                'defensive_success_rate': 0.40,
                'average_score': 0.3
            },
            'trained_ppo_baseline': {
                'overall_success_rate': 0.45,
                'bvr_success_rate': 0.50,
                'wvr_success_rate': 0.35,
                'defensive_success_rate': 0.60,
                'average_score': 0.5
            },
            'llm_guided_agent': {
                'overall_success_rate': 0.65,
                'bvr_success_rate': 0.70,
                'wvr_success_rate': 0.55,
                'defensive_success_rate': 0.75,
                'average_score': 0.7
            },
            'expert_agent': {
                'overall_success_rate': 0.85,
                'bvr_success_rate': 0.90,
                'wvr_success_rate': 0.75,
                'defensive_success_rate': 0.90,
                'average_score': 0.9
            }
        }
    
    def _print_scenario_overview(self):
        """Print overview of evaluation scenarios"""
        
        print(f"\n{'='*80}")
        print("COMBAT EVALUATION SCENARIOS")
        print(f"{'='*80}")
        
        for scenario_type, config in self.scenarios.items():
            print(f"\n🎯 {config.name}")
            print(f"   Description: {config.description}")
            print(f"   Difficulty: {config.difficulty_level:.1f}/1.0")
            print(f"   Expected Duration: {config.expected_duration} steps")
            print(f"   Focus Areas: {', '.join(config.tactical_focus)}")
        
        print(f"\n{'='*80}")
    
    def evaluate_agent(self, agent, env, num_episodes_per_scenario: int = 20,
                      scenarios_to_test: List[EvaluationScenario] = None) -> Dict[str, Any]:
        """
        Comprehensive agent evaluation across all scenarios
        
        Args:
            agent: RL agent to evaluate
            env: Environment for evaluation
            num_episodes_per_scenario: Episodes per scenario
            scenarios_to_test: Specific scenarios to test (all if None)
        
        Returns:
            Comprehensive evaluation results
        """
        
        if scenarios_to_test is None:
            scenarios_to_test = list(self.scenarios.keys())
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE AGENT EVALUATION")
        print(f"{'='*80}")
        print(f"Scenarios: {len(scenarios_to_test)}")
        print(f"Episodes per scenario: {num_episodes_per_scenario}")
        print(f"Total episodes: {len(scenarios_to_test) * num_episodes_per_scenario}")
        
        evaluation_results = {
            'agent_id': f"agent_{int(time.time())}",
            'evaluation_timestamp': time.time(),
            'scenario_results': {},
            'overall_performance': {},
            'benchmark_comparison': {},
            'recommendations': []
        }
        
        all_results = []
        
        # Evaluate each scenario
        for scenario_type in scenarios_to_test:
            scenario_config = self.scenarios[scenario_type]
            
            print(f"\n[EVAL] Testing {scenario_config.name}...")
            
            scenario_results = self._evaluate_scenario(
                agent, env, scenario_type, num_episodes_per_scenario
            )
            
            evaluation_results['scenario_results'][scenario_type.value] = scenario_results
            all_results.extend(scenario_results['episode_results'])
            
            print(f"[EVAL] {scenario_config.name}: "
                  f"Success {scenario_results['success_rate']:.1%}, "
                  f"Score {scenario_results['average_score']:.2f}")
        
        # Calculate overall performance
        evaluation_results['overall_performance'] = self._calculate_overall_performance(all_results)
        
        # Compare against benchmarks
        evaluation_results['benchmark_comparison'] = self._compare_against_benchmarks(
            evaluation_results['overall_performance']
        )
        
        # Generate recommendations
        evaluation_results['recommendations'] = self._generate_evaluation_recommendations(
            evaluation_results
        )
        
        # Store evaluation history
        self.evaluation_history.append(evaluation_results)
        
        # Print summary
        self._print_evaluation_summary(evaluation_results)
        
        return evaluation_results
    
    def _evaluate_scenario(self, agent, env, scenario_type: EvaluationScenario,
                          num_episodes: int) -> Dict[str, Any]:
        """Evaluate agent on specific scenario"""
        
        scenario_config = self.scenarios[scenario_type]
        episode_results = []
        
        for episode in range(num_episodes):
            # Configure environment for scenario
            if hasattr(env, 'set_scenario_config'):
                env.set_scenario_config(scenario_config.initial_conditions)
            
            # Run episode
            result = self._run_evaluation_episode(agent, env, scenario_config, episode)
            episode_results.append(result)
            
            if self.verbose and episode % 5 == 0:
                print(f"   Episode {episode+1}/{num_episodes}: "
                      f"Success={result.success}, Score={result.score:.2f}")
        
        # Calculate scenario statistics
        success_rate = np.mean([r.success for r in episode_results])
        average_score = np.mean([r.score for r in episode_results])
        average_duration = np.mean([r.duration for r in episode_results])
        
        # Tactical performance analysis
        tactical_performance = self._analyze_tactical_performance(episode_results)
        
        return {
            'scenario_name': scenario_config.name,
            'episodes_completed': len(episode_results),
            'success_rate': success_rate,
            'average_score': average_score,
            'average_duration': average_duration,
            'tactical_performance': tactical_performance,
            'episode_results': episode_results
        }
    
    def _run_evaluation_episode(self, agent, env, scenario_config: ScenarioConfig,
                               episode_id: int) -> EvaluationResult:
        """Run single evaluation episode"""
        
        # Reset environment
        obs, info = env.reset(seed=episode_id) if hasattr(env, 'reset') else (env.reset(), {})
        
        done = False
        step = 0
        total_reward = 0
        action_history = []
        state_history = []
        
        # Episode metrics
        episode_metrics = {
            'shots_fired': 0,
            'shots_hit': 0,
            'lock_duration': 0,
            'max_threat_level': 0,
            'energy_violations': 0,
            'tactical_errors': 0
        }
        
        while not done and step < scenario_config.expected_duration * 2:  # 2x safety margin
            # Agent action
            if hasattr(agent, 'predict'):
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()  # Fallback for testing
            
            # Environment step
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
                info = info if isinstance(info, dict) else {}
            
            # Record data
            total_reward += reward
            action_history.append(action.copy() if hasattr(action, 'copy') else action)
            state_history.append(obs.copy() if hasattr(obs, 'copy') else obs)
            
            # Update episode metrics
            self._update_episode_metrics(episode_metrics, action, info)
            
            step += 1
        
        # Evaluate episode performance
        success = self._check_success_criteria(scenario_config.success_criteria, 
                                             episode_metrics, info, step)
        
        score = self._calculate_episode_score(scenario_config, episode_metrics, 
                                            total_reward, step, success)
        
        # Tactical analysis
        tactical_metrics = self._analyze_episode_tactics(action_history, state_history, 
                                                       episode_metrics)
        
        # Performance breakdown
        performance_breakdown = self._calculate_performance_breakdown(
            scenario_config, episode_metrics, tactical_metrics
        )
        
        # Generate recommendations
        recommendations = self._generate_episode_recommendations(
            scenario_config, episode_metrics, tactical_metrics
        )
        
        return EvaluationResult(
            scenario_name=scenario_config.name,
            success=success,
            score=score,
            duration=step,
            tactical_metrics=tactical_metrics,
            performance_breakdown=performance_breakdown,
            recommendations=recommendations,
            detailed_log={
                'episode_metrics': episode_metrics,
                'total_reward': total_reward,
                'final_info': info
            }
        )
    
    def _update_episode_metrics(self, metrics: Dict[str, Any], action: np.ndarray, 
                               info: Dict[str, Any]):
        """Update episode metrics during evaluation"""
        
        # Shot tracking
        if hasattr(action, '__len__') and len(action) > 3 and action[3] > 0.5:
            metrics['shots_fired'] += 1
            if info.get('hit', False):
                metrics['shots_hit'] += 1
        
        # Lock tracking
        if info.get('locked', False):
            metrics['lock_duration'] += 1
        
        # Threat tracking
        threat_level = info.get('threat_level', 0)
        metrics['max_threat_level'] = max(metrics['max_threat_level'], threat_level)
        
        # Energy violations
        if info.get('energy_state') == 'LOW' and info.get('engagement_phase') == 'BVR':
            metrics['energy_violations'] += 1
    
    def _check_success_criteria(self, criteria: Dict[str, float], 
                               episode_metrics: Dict[str, Any],
                               final_info: Dict[str, Any], duration: int) -> bool:
        """Check if episode met success criteria"""
        
        success = True
        
        # Check each criterion
        if 'enemy_destroyed' in criteria:
            enemy_health = final_info.get('enemy_health', 1.0)
            if enemy_health > (1.0 - criteria['enemy_destroyed']):
                success = False
        
        if 'survival' in criteria:
            ego_health = final_info.get('ego_health', 1.0)
            if ego_health <= (1.0 - criteria['survival']):
                success = False
        
        if 'time_limit' in criteria:
            if duration > criteria['time_limit']:
                success = False
        
        if 'missile_efficiency' in criteria:
            shots_fired = episode_metrics.get('shots_fired', 0)
            shots_hit = episode_metrics.get('shots_hit', 0)
            efficiency = shots_hit / max(shots_fired, 1)
            if efficiency < criteria['missile_efficiency']:
                success = False
        
        return success
    
    def _calculate_episode_score(self, scenario_config: ScenarioConfig,
                               episode_metrics: Dict[str, Any],
                               total_reward: float, duration: int, success: bool) -> float:
        """Calculate comprehensive episode score"""
        
        base_score = 1.0 if success else 0.0
        
        # Time efficiency bonus
        expected_duration = scenario_config.expected_duration
        if duration <= expected_duration:
            time_bonus = 0.2 * (1.0 - duration / expected_duration)
        else:
            time_bonus = -0.1 * ((duration - expected_duration) / expected_duration)
        
        # Tactical efficiency
        shots_fired = episode_metrics.get('shots_fired', 0)
        shots_hit = episode_metrics.get('shots_hit', 0)
        shot_accuracy = shots_hit / max(shots_fired, 1)
        accuracy_bonus = 0.1 * shot_accuracy
        
        # Lock efficiency
        lock_duration = episode_metrics.get('lock_duration', 0)
        lock_efficiency = min(1.0, lock_duration / max(duration, 1))
        lock_bonus = 0.1 * lock_efficiency
        
        # Energy management
        energy_violations = episode_metrics.get('energy_violations', 0)
        energy_penalty = -0.05 * energy_violations
        
        # Total score
        total_score = base_score + time_bonus + accuracy_bonus + lock_bonus + energy_penalty
        
        return max(0.0, min(1.0, total_score))
    
    def _analyze_tactical_performance(self, episode_results: List[EvaluationResult]) -> Dict[str, float]:
        """Analyze tactical performance across episodes"""
        
        if not episode_results:
            return {}
        
        # Aggregate tactical metrics
        all_tactical_metrics = [r.tactical_metrics for r in episode_results]
        
        aggregated = {}
        for metrics in all_tactical_metrics:
            for key, value in metrics.items():
                if key not in aggregated:
                    aggregated[key] = []
                aggregated[key].append(value)
        
        # Calculate averages
        tactical_performance = {}
        for key, values in aggregated.items():
            tactical_performance[f"avg_{key}"] = np.mean(values)
            tactical_performance[f"std_{key}"] = np.std(values)
        
        return tactical_performance
    
    def _analyze_episode_tactics(self, action_history: List[np.ndarray],
                               state_history: List[np.ndarray],
                               episode_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Analyze tactical performance for single episode"""
        
        tactics = {}
        
        # Action smoothness
        if len(action_history) > 1:
            action_changes = []
            for i in range(1, len(action_history)):
                if hasattr(action_history[i], '__len__') and len(action_history[i]) > 3:
                    change = np.linalg.norm(action_history[i][:3] - action_history[i-1][:3])
                    action_changes.append(change)
            
            tactics['action_smoothness'] = 1.0 / (1.0 + np.mean(action_changes)) if action_changes else 1.0
        else:
            tactics['action_smoothness'] = 1.0
        
        # Shot accuracy
        shots_fired = episode_metrics.get('shots_fired', 0)
        shots_hit = episode_metrics.get('shots_hit', 0)
        tactics['shot_accuracy'] = shots_hit / max(shots_fired, 1)
        
        # Lock efficiency
        lock_duration = episode_metrics.get('lock_duration', 0)
        total_steps = len(action_history)
        tactics['lock_efficiency'] = lock_duration / max(total_steps, 1)
        
        # Threat response
        max_threat = episode_metrics.get('max_threat_level', 0)
        tactics['threat_awareness'] = min(1.0, max_threat)
        
        # Energy management
        energy_violations = episode_metrics.get('energy_violations', 0)
        tactics['energy_management'] = max(0.0, 1.0 - energy_violations / 10.0)
        
        return tactics
    
    def _calculate_performance_breakdown(self, scenario_config: ScenarioConfig,
                                       episode_metrics: Dict[str, Any],
                                       tactical_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate detailed performance breakdown"""
        
        breakdown = {}
        
        # Focus area performance
        for focus_area in scenario_config.tactical_focus:
            if focus_area == 'bvr_tactics':
                breakdown[focus_area] = tactical_metrics.get('shot_accuracy', 0) * 0.6 + \
                                      tactical_metrics.get('lock_efficiency', 0) * 0.4
            elif focus_area == 'wvr_maneuvering':
                breakdown[focus_area] = tactical_metrics.get('action_smoothness', 0) * 0.5 + \
                                      tactical_metrics.get('energy_management', 0) * 0.5
            elif focus_area == 'defensive_bfm':
                breakdown[focus_area] = tactical_metrics.get('threat_awareness', 0) * 0.7 + \
                                      tactical_metrics.get('energy_management', 0) * 0.3
            else:
                # Generic performance
                breakdown[focus_area] = np.mean(list(tactical_metrics.values()))
        
        return breakdown
    
    def _generate_episode_recommendations(self, scenario_config: ScenarioConfig,
                                        episode_metrics: Dict[str, Any],
                                        tactical_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations for episode performance"""
        
        recommendations = []
        
        # Shot accuracy recommendations
        if tactical_metrics.get('shot_accuracy', 1.0) < 0.3:
            recommendations.append("Improve shot timing - ensure proper lock before firing")
        
        # Lock efficiency recommendations
        if tactical_metrics.get('lock_efficiency', 1.0) < 0.4:
            recommendations.append("Focus on target tracking - maintain radar lock longer")
        
        # Energy management recommendations
        if tactical_metrics.get('energy_management', 1.0) < 0.6:
            recommendations.append("Improve energy management - avoid low energy in BVR")
        
        # Action smoothness recommendations
        if tactical_metrics.get('action_smoothness', 1.0) < 0.7:
            recommendations.append("Smooth control inputs - avoid excessive maneuvering")
        
        # Scenario-specific recommendations
        if scenario_config.name == "Head-On BVR Engagement":
            if episode_metrics.get('shots_fired', 0) == 0:
                recommendations.append("BVR engagement requires missile employment")
        
        elif scenario_config.name == "Defensive BFM":
            if tactical_metrics.get('threat_awareness', 0) < 0.8:
                recommendations.append("Improve threat recognition and defensive timing")
        
        return recommendations
    
    def _calculate_overall_performance(self, all_results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate overall performance across all scenarios"""
        
        if not all_results:
            return {}
        
        return {
            'overall_success_rate': np.mean([r.success for r in all_results]),
            'overall_average_score': np.mean([r.score for r in all_results]),
            'average_duration': np.mean([r.duration for r in all_results]),
            'episodes_evaluated': len(all_results),
            'performance_consistency': 1.0 - np.std([r.score for r in all_results])
        }
    
    def _compare_against_benchmarks(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Compare performance against established benchmarks"""
        
        overall_success = performance.get('overall_success_rate', 0)
        
        # Find closest benchmark
        best_match = 'random_agent'
        best_diff = float('inf')
        
        for benchmark_name, benchmark_data in self.benchmarks.items():
            diff = abs(benchmark_data['overall_success_rate'] - overall_success)
            if diff < best_diff:
                best_diff = diff
                best_match = benchmark_name
        
        # Performance level assessment
        if overall_success >= 0.8:
            performance_level = "Expert"
        elif overall_success >= 0.6:
            performance_level = "Advanced"
        elif overall_success >= 0.4:
            performance_level = "Intermediate"
        elif overall_success >= 0.2:
            performance_level = "Novice"
        else:
            performance_level = "Beginner"
        
        return {
            'closest_benchmark': best_match,
            'performance_level': performance_level,
            'benchmark_data': self.benchmarks[best_match],
            'performance_gap': overall_success - self.benchmarks[best_match]['overall_success_rate']
        }
    
    def _generate_evaluation_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations based on evaluation"""
        
        recommendations = []
        
        overall_perf = evaluation_results['overall_performance']
        benchmark_comp = evaluation_results['benchmark_comparison']
        
        # Overall performance recommendations
        success_rate = overall_perf.get('overall_success_rate', 0)
        
        if success_rate < 0.3:
            recommendations.append("Focus on basic combat fundamentals - consider curriculum learning")
        elif success_rate < 0.5:
            recommendations.append("Improve tactical decision making - increase LLM guidance frequency")
        elif success_rate < 0.7:
            recommendations.append("Refine advanced tactics - focus on specific scenario weaknesses")
        else:
            recommendations.append("Excellent performance - ready for advanced training scenarios")
        
        # Benchmark-based recommendations
        performance_level = benchmark_comp['performance_level']
        if performance_level in ['Beginner', 'Novice']:
            recommendations.append("Consider longer training duration or curriculum learning")
        elif performance_level == 'Intermediate':
            recommendations.append("Focus on scenario-specific training to reach advanced level")
        
        return recommendations
    
    def _print_evaluation_summary(self, evaluation_results: Dict[str, Any]):
        """Print comprehensive evaluation summary"""
        
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        overall_perf = evaluation_results['overall_performance']
        benchmark_comp = evaluation_results['benchmark_comparison']
        
        print(f"Overall Success Rate: {overall_perf.get('overall_success_rate', 0):.1%}")
        print(f"Average Score: {overall_perf.get('overall_average_score', 0):.2f}")
        print(f"Performance Level: {benchmark_comp['performance_level']}")
        print(f"Closest Benchmark: {benchmark_comp['closest_benchmark']}")
        
        print(f"\nScenario Performance:")
        for scenario_name, results in evaluation_results['scenario_results'].items():
            print(f"  {results['scenario_name']}: {results['success_rate']:.1%} success")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(evaluation_results['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print(f"{'='*80}")
    
    def export_evaluation_report(self, evaluation_results: Dict[str, Any],
                               filepath: str = None) -> str:
        """Export detailed evaluation report"""
        
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"evaluation/reports/combat_evaluation_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        print(f"[EVAL SUITE] Evaluation report exported: {filepath}")
        return filepath
    
    def compare_agents(self, agents: Dict[str, Any], env, 
                      num_episodes_per_scenario: int = 10) -> Dict[str, Any]:
        """Compare multiple agents across all scenarios"""
        
        print(f"\n{'='*80}")
        print("MULTI-AGENT COMPARISON")
        print(f"{'='*80}")
        
        comparison_results = {
            'comparison_timestamp': time.time(),
            'agents_compared': list(agents.keys()),
            'agent_results': {},
            'comparative_analysis': {}
        }
        
        # Evaluate each agent
        for agent_name, agent in agents.items():
            print(f"\n[COMPARISON] Evaluating {agent_name}...")
            
            agent_results = self.evaluate_agent(
                agent, env, num_episodes_per_scenario, 
                scenarios_to_test=list(self.scenarios.keys())[:5]  # Test subset for comparison
            )
            
            comparison_results['agent_results'][agent_name] = agent_results
        
        # Comparative analysis
        comparison_results['comparative_analysis'] = self._analyze_agent_comparison(
            comparison_results['agent_results']
        )
        
        return comparison_results
    
    def _analyze_agent_comparison(self, agent_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze comparison between multiple agents"""
        
        if len(agent_results) < 2:
            return {'error': 'Need at least 2 agents for comparison'}
        
        # Extract performance metrics
        agent_performances = {}
        for agent_name, results in agent_results.items():
            overall_perf = results['overall_performance']
            agent_performances[agent_name] = overall_perf.get('overall_success_rate', 0)
        
        # Rank agents
        ranked_agents = sorted(agent_performances.items(), key=lambda x: x[1], reverse=True)
        
        analysis = {
            'agent_ranking': ranked_agents,
            'best_agent': ranked_agents[0][0],
            'worst_agent': ranked_agents[-1][0],
            'performance_spread': ranked_agents[0][1] - ranked_agents[-1][1],
            'relative_improvements': {}
        }
        
        # Calculate relative improvements
        baseline_performance = ranked_agents[-1][1]  # Worst agent as baseline
        for agent_name, performance in ranked_agents:
            if baseline_performance > 0:
                improvement = (performance - baseline_performance) / baseline_performance
                analysis['relative_improvements'][agent_name] = improvement
        
        return analysis


def create_comprehensive_evaluation_system(llm_assistant):
    """
    Create complete evaluation system with combat scenarios and LLM analysis
    
    Args:
        llm_assistant: LLM assistant to analyze
    
    Returns:
        (combat_eval_suite, llm_analyzer)
    """
    
    print("[EVAL SYSTEM] Creating comprehensive evaluation system...")
    
    # Import here to avoid circular imports
    from analysis.llm_effectiveness_analyzer import LLMEffectivenessAnalyzer
    
    # Create components
    combat_eval_suite = CombatEvaluationSuite(verbose=True)
    llm_analyzer = LLMEffectivenessAnalyzer(analysis_window=1000, verbose=True)
    
    print("[EVAL SYSTEM] Comprehensive evaluation system ready")
    print(f"   Combat scenarios: {len(combat_eval_suite.scenarios)}")
    print(f"   LLM analysis window: {llm_analyzer.analysis_window}")
    
    return combat_eval_suite, llm_analyzer


if __name__ == "__main__":
    # Example usage
    print("Comprehensive Combat Evaluation Suite")
    
    # Create evaluation suite
    eval_suite = CombatEvaluationSuite(verbose=True)
    
    print(f"\nEvaluation suite ready with {len(eval_suite.scenarios)} scenarios")
    print("Use evaluate_agent() to test RL agents against standardized scenarios")
