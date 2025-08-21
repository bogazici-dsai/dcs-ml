# LLM Effectiveness Analyzer for Combat Training Optimization
import numpy as np
import json
import time
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque

# Optional visualization dependencies
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


@dataclass
class LLMInteractionRecord:
    """Record of LLM interaction for analysis"""
    timestamp: float
    step: int
    episode: int
    tactical_situation: str
    llm_prompt: str
    llm_response: str
    shaping_delta: float
    response_time: float
    agent_action_before: np.ndarray
    agent_action_after: np.ndarray
    environment_reward: float
    success_outcome: bool


class LLMEffectivenessAnalyzer:
    """
    Advanced analytics system for measuring and optimizing LLM guidance effectiveness
    in combat training. Analyzes when, how, and why LLM interventions help or hurt performance.
    """
    
    def __init__(self, analysis_window: int = 1000, verbose: bool = True):
        """
        Initialize LLM effectiveness analyzer
        
        Args:
            analysis_window: Number of recent interactions to analyze
            verbose: Enable detailed logging
        """
        self.analysis_window = analysis_window
        self.verbose = verbose
        
        # Data storage
        self.interaction_history = deque(maxlen=analysis_window)
        self.episode_summaries = []
        self.performance_trends = defaultdict(list)
        
        # Analysis results
        self.effectiveness_metrics = {}
        self.intervention_patterns = {}
        self.guidance_quality_scores = {}
        
        print(f"[LLM ANALYZER] Initialized with {analysis_window} interaction window")
    
    def record_llm_interaction(self, step: int, episode: int, tactical_situation: str,
                             llm_prompt: str, llm_response: str, shaping_delta: float,
                             response_time: float, agent_action_before: np.ndarray,
                             agent_action_after: np.ndarray, environment_reward: float,
                             success_outcome: bool = False):
        """
        Record LLM interaction for analysis
        
        Args:
            step: Training step
            episode: Episode number
            tactical_situation: Description of tactical situation
            llm_prompt: Prompt sent to LLM
            llm_response: LLM response text
            shaping_delta: Reward shaping value
            response_time: LLM response time
            agent_action_before: Agent action before LLM guidance
            agent_action_after: Final action after LLM guidance
            environment_reward: Base environment reward
            success_outcome: Whether the action led to success
        """
        
        record = LLMInteractionRecord(
            timestamp=time.time(),
            step=step,
            episode=episode,
            tactical_situation=tactical_situation,
            llm_prompt=llm_prompt,
            llm_response=llm_response,
            shaping_delta=shaping_delta,
            response_time=response_time,
            agent_action_before=agent_action_before.copy(),
            agent_action_after=agent_action_after.copy(),
            environment_reward=environment_reward,
            success_outcome=success_outcome
        )
        
        self.interaction_history.append(record)
        
        # Trigger analysis every 100 interactions
        if len(self.interaction_history) % 100 == 0:
            self._update_running_analysis()
    
    def analyze_llm_impact(self, episode_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive analysis of LLM impact on training performance
        
        Args:
            episode_logs: List of episode log data
        
        Returns:
            Comprehensive LLM impact analysis
        """
        
        print(f"[LLM ANALYZER] Analyzing {len(episode_logs)} episodes...")
        
        analysis = {
            'guidance_accuracy': self._measure_guidance_accuracy(episode_logs),
            'intervention_timing': self._analyze_intervention_timing(episode_logs),
            'tactical_knowledge_gaps': self._identify_knowledge_gaps(episode_logs),
            'reward_shaping_effectiveness': self._measure_shaping_impact(episode_logs),
            'response_quality': self._analyze_response_quality(episode_logs),
            'situational_effectiveness': self._analyze_situational_effectiveness(episode_logs)
        }
        
        # Generate optimization recommendations
        analysis['optimization_recommendations'] = self._generate_optimization_recommendations(analysis)
        
        return analysis
    
    def _measure_guidance_accuracy(self, episode_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Measure accuracy of LLM tactical guidance"""
        
        if not self.interaction_history:
            return {'error': 'No interaction data'}
        
        # Analyze guidance vs outcomes
        correct_guidance = 0
        total_guidance = 0
        
        for record in self.interaction_history:
            total_guidance += 1
            
            # Define "correct" guidance based on outcome
            if record.success_outcome and record.shaping_delta > 0:
                correct_guidance += 1  # Positive shaping led to success
            elif not record.success_outcome and record.shaping_delta < 0:
                correct_guidance += 1  # Negative shaping correctly identified poor action
            elif abs(record.shaping_delta) < 0.1 and record.environment_reward > 0:
                correct_guidance += 1  # Neutral shaping for good base action
        
        accuracy = correct_guidance / max(total_guidance, 1)
        
        # Analyze by tactical situation
        situation_accuracy = defaultdict(list)
        for record in self.interaction_history:
            situation = record.tactical_situation
            correct = (record.success_outcome and record.shaping_delta > 0) or \
                     (not record.success_outcome and record.shaping_delta < 0)
            situation_accuracy[situation].append(correct)
        
        situation_scores = {situation: np.mean(scores) 
                          for situation, scores in situation_accuracy.items()}
        
        return {
            'overall_accuracy': accuracy,
            'situation_specific_accuracy': situation_scores,
            'total_interactions_analyzed': total_guidance,
            'correct_predictions': correct_guidance
        }
    
    def _analyze_intervention_timing(self, episode_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze optimal timing for LLM interventions"""
        
        # Analyze when LLM interventions are most/least effective
        timing_effectiveness = defaultdict(list)
        
        for record in self.interaction_history:
            # Categorize timing
            if record.step < 50:
                timing_category = 'early_episode'
            elif record.step < 200:
                timing_category = 'mid_episode'
            else:
                timing_category = 'late_episode'
            
            # Measure effectiveness
            effectiveness = self._calculate_intervention_effectiveness(record)
            timing_effectiveness[timing_category].append(effectiveness)
        
        # Calculate timing statistics
        timing_stats = {}
        for category, effectiveness_scores in timing_effectiveness.items():
            timing_stats[category] = {
                'average_effectiveness': np.mean(effectiveness_scores),
                'intervention_count': len(effectiveness_scores),
                'success_rate': np.mean([e > 0.5 for e in effectiveness_scores])
            }
        
        # Find optimal intervention frequency
        optimal_frequency = self._find_optimal_intervention_frequency()
        
        return {
            'timing_effectiveness': timing_stats,
            'optimal_intervention_frequency': optimal_frequency,
            'current_intervention_rate': len(self.interaction_history) / max(1, self.interaction_history[-1].step if self.interaction_history else 1)
        }
    
    def _identify_knowledge_gaps(self, episode_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify areas where LLM lacks combat knowledge"""
        
        knowledge_gaps = {
            'tactical_situations_with_poor_guidance': [],
            'consistently_wrong_recommendations': [],
            'knowledge_gap_categories': defaultdict(int),
            'improvement_opportunities': []
        }
        
        # Analyze situations where LLM guidance was consistently poor
        situation_performance = defaultdict(list)
        
        for record in self.interaction_history:
            effectiveness = self._calculate_intervention_effectiveness(record)
            situation_performance[record.tactical_situation].append(effectiveness)
        
        # Identify poor performance situations
        for situation, effectiveness_scores in situation_performance.items():
            avg_effectiveness = np.mean(effectiveness_scores)
            if avg_effectiveness < 0.3 and len(effectiveness_scores) >= 5:
                knowledge_gaps['tactical_situations_with_poor_guidance'].append({
                    'situation': situation,
                    'average_effectiveness': avg_effectiveness,
                    'sample_count': len(effectiveness_scores)
                })
        
        # Categorize knowledge gaps
        for gap in knowledge_gaps['tactical_situations_with_poor_guidance']:
            situation = gap['situation'].lower()
            if 'bvr' in situation:
                knowledge_gaps['knowledge_gap_categories']['bvr_tactics'] += 1
            elif 'wvr' in situation or 'dogfight' in situation:
                knowledge_gaps['knowledge_gap_categories']['wvr_tactics'] += 1
            elif 'defensive' in situation:
                knowledge_gaps['knowledge_gap_categories']['defensive_tactics'] += 1
            elif 'energy' in situation:
                knowledge_gaps['knowledge_gap_categories']['energy_management'] += 1
        
        # Generate improvement opportunities
        for category, gap_count in knowledge_gaps['knowledge_gap_categories'].items():
            if gap_count >= 2:
                knowledge_gaps['improvement_opportunities'].append(
                    f"Focus fine-tuning on {category} - {gap_count} knowledge gaps identified"
                )
        
        return knowledge_gaps
    
    def _measure_shaping_impact(self, episode_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Measure impact of reward shaping on learning"""
        
        if not self.interaction_history:
            return {'error': 'No interaction data'}
        
        # Analyze shaping delta distribution
        shaping_deltas = [record.shaping_delta for record in self.interaction_history]
        
        # Analyze correlation with success
        positive_shaping_success = []
        negative_shaping_success = []
        neutral_shaping_success = []
        
        for record in self.interaction_history:
            if record.shaping_delta > 0.1:
                positive_shaping_success.append(record.success_outcome)
            elif record.shaping_delta < -0.1:
                negative_shaping_success.append(record.success_outcome)
            else:
                neutral_shaping_success.append(record.success_outcome)
        
        return {
            'average_shaping_delta': np.mean(shaping_deltas),
            'shaping_delta_std': np.std(shaping_deltas),
            'positive_shaping_success_rate': np.mean(positive_shaping_success) if positive_shaping_success else 0,
            'negative_shaping_success_rate': np.mean(negative_shaping_success) if negative_shaping_success else 0,
            'neutral_shaping_success_rate': np.mean(neutral_shaping_success) if neutral_shaping_success else 0,
            'shaping_correlation_with_success': self._calculate_shaping_correlation()
        }
    
    def _analyze_response_quality(self, episode_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze quality of LLM responses"""
        
        if not self.interaction_history:
            return {'error': 'No interaction data'}
        
        # Response time analysis
        response_times = [record.response_time for record in self.interaction_history]
        
        # Response length analysis
        response_lengths = [len(record.llm_response) for record in self.interaction_history]
        
        # Parse quality indicators from responses
        quality_indicators = {
            'contains_tactical_terms': 0,
            'contains_specific_recommendations': 0,
            'contains_reasoning': 0,
            'parseable_json': 0
        }
        
        for record in self.interaction_history:
            response = record.llm_response.lower()
            
            # Check for tactical terminology
            tactical_terms = ['bvr', 'wvr', 'missile', 'radar', 'lock', 'maneuver', 'energy', 'threat']
            if any(term in response for term in tactical_terms):
                quality_indicators['contains_tactical_terms'] += 1
            
            # Check for specific recommendations
            if any(word in response for word in ['fire', 'turn', 'climb', 'dive', 'evade']):
                quality_indicators['contains_specific_recommendations'] += 1
            
            # Check for reasoning
            if any(word in response for word in ['because', 'since', 'due to', 'reason']):
                quality_indicators['contains_reasoning'] += 1
            
            # Check for JSON parseability
            try:
                json.loads(response)
                quality_indicators['parseable_json'] += 1
            except:
                pass
        
        # Convert to percentages
        total_interactions = len(self.interaction_history)
        quality_percentages = {key: (count / max(total_interactions, 1)) 
                             for key, count in quality_indicators.items()}
        
        return {
            'average_response_time': np.mean(response_times),
            'response_time_std': np.std(response_times),
            'average_response_length': np.mean(response_lengths),
            'response_quality_indicators': quality_percentages,
            'overall_response_quality': np.mean(list(quality_percentages.values()))
        }
    
    def _analyze_situational_effectiveness(self, episode_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze LLM effectiveness in different tactical situations"""
        
        situational_analysis = defaultdict(lambda: {
            'interactions': 0,
            'average_effectiveness': 0.0,
            'success_rate': 0.0,
            'average_shaping': 0.0,
            'response_quality': 0.0
        })
        
        # Group by tactical situation
        for record in self.interaction_history:
            situation = record.tactical_situation
            effectiveness = self._calculate_intervention_effectiveness(record)
            
            analysis = situational_analysis[situation]
            analysis['interactions'] += 1
            analysis['average_effectiveness'] = (
                analysis['average_effectiveness'] * (analysis['interactions'] - 1) + effectiveness
            ) / analysis['interactions']
            analysis['success_rate'] = (
                analysis['success_rate'] * (analysis['interactions'] - 1) + record.success_outcome
            ) / analysis['interactions']
            analysis['average_shaping'] = (
                analysis['average_shaping'] * (analysis['interactions'] - 1) + record.shaping_delta
            ) / analysis['interactions']
        
        # Convert to regular dict and sort by effectiveness
        sorted_situations = sorted(
            situational_analysis.items(),
            key=lambda x: x[1]['average_effectiveness'],
            reverse=True
        )
        
        return {
            'situational_effectiveness': dict(sorted_situations),
            'best_situations': sorted_situations[:3],
            'worst_situations': sorted_situations[-3:],
            'total_situations_analyzed': len(situational_analysis)
        }
    
    def _calculate_intervention_effectiveness(self, record: LLMInteractionRecord) -> float:
        """Calculate effectiveness of single LLM intervention"""
        
        # Base effectiveness from outcome
        outcome_effectiveness = 1.0 if record.success_outcome else 0.0
        
        # Shaping appropriateness
        shaping_appropriateness = 0.5  # Neutral
        if record.success_outcome and record.shaping_delta > 0:
            shaping_appropriateness = 1.0  # Correctly positive
        elif not record.success_outcome and record.shaping_delta < 0:
            shaping_appropriateness = 1.0  # Correctly negative
        elif record.success_outcome and record.shaping_delta < -0.2:
            shaping_appropriateness = 0.0  # Incorrectly negative
        elif not record.success_outcome and record.shaping_delta > 0.2:
            shaping_appropriateness = 0.0  # Incorrectly positive
        
        # Action modification appropriateness
        action_change = np.linalg.norm(record.agent_action_after - record.agent_action_before)
        action_effectiveness = 0.5
        
        if record.success_outcome and action_change > 0.2:
            action_effectiveness = 0.8  # Good modification
        elif not record.success_outcome and action_change < 0.1:
            action_effectiveness = 0.2  # Should have modified more
        
        # Response time factor
        time_factor = min(1.0, 2.0 / max(record.response_time, 0.1))  # Prefer <2s responses
        
        # Combined effectiveness
        effectiveness = (
            outcome_effectiveness * 0.4 +
            shaping_appropriateness * 0.3 +
            action_effectiveness * 0.2 +
            time_factor * 0.1
        )
        
        return effectiveness
    
    def _find_optimal_intervention_frequency(self) -> float:
        """Find optimal LLM intervention frequency"""
        
        if len(self.interaction_history) < 50:
            return 10.0  # Default frequency
        
        # Analyze effectiveness vs frequency
        # Group interactions by frequency bands
        frequency_bands = {
            'very_high': [],  # >20 Hz
            'high': [],       # 10-20 Hz
            'medium': [],     # 5-10 Hz
            'low': [],        # 1-5 Hz
            'very_low': []    # <1 Hz
        }
        
        # Calculate local frequency for each interaction
        for i, record in enumerate(self.interaction_history):
            if i < 10:
                continue  # Need history for frequency calculation
            
            # Calculate frequency over last 10 interactions
            time_window = record.timestamp - self.interaction_history[i-10].timestamp
            local_frequency = 10.0 / max(time_window, 1.0)
            
            effectiveness = self._calculate_intervention_effectiveness(record)
            
            # Categorize frequency
            if local_frequency > 20:
                frequency_bands['very_high'].append(effectiveness)
            elif local_frequency > 10:
                frequency_bands['high'].append(effectiveness)
            elif local_frequency > 5:
                frequency_bands['medium'].append(effectiveness)
            elif local_frequency > 1:
                frequency_bands['low'].append(effectiveness)
            else:
                frequency_bands['very_low'].append(effectiveness)
        
        # Find frequency band with highest effectiveness
        best_frequency = 10.0  # Default
        best_effectiveness = 0.0
        
        frequency_mapping = {
            'very_high': 25.0,
            'high': 15.0,
            'medium': 7.5,
            'low': 3.0,
            'very_low': 0.5
        }
        
        for band, effectiveness_scores in frequency_bands.items():
            if effectiveness_scores:
                avg_effectiveness = np.mean(effectiveness_scores)
                if avg_effectiveness > best_effectiveness:
                    best_effectiveness = avg_effectiveness
                    best_frequency = frequency_mapping[band]
        
        return best_frequency
    
    def _calculate_shaping_correlation(self) -> float:
        """Calculate correlation between shaping delta and success"""
        
        if len(self.interaction_history) < 10:
            return 0.0
        
        shaping_deltas = [record.shaping_delta for record in self.interaction_history]
        success_outcomes = [1.0 if record.success_outcome else 0.0 for record in self.interaction_history]
        
        # Calculate Pearson correlation
        correlation = np.corrcoef(shaping_deltas, success_outcomes)[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def _update_running_analysis(self):
        """Update running analysis with latest data"""
        
        if len(self.interaction_history) < 50:
            return
        
        # Update effectiveness metrics
        recent_interactions = list(self.interaction_history)[-50:]
        recent_effectiveness = [self._calculate_intervention_effectiveness(record) 
                              for record in recent_interactions]
        
        self.effectiveness_metrics['recent_average'] = np.mean(recent_effectiveness)
        self.effectiveness_metrics['recent_std'] = np.std(recent_effectiveness)
        
        # Update intervention patterns
        self.intervention_patterns['recent_frequency'] = len(recent_interactions) / 50.0
        self.intervention_patterns['recent_success_rate'] = np.mean([r.success_outcome for r in recent_interactions])
        
        if self.verbose and len(self.interaction_history) % 200 == 0:
            print(f"[LLM ANALYZER] Recent effectiveness: {self.effectiveness_metrics['recent_average']:.2f}")
    
    def _generate_optimization_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for optimizing LLM guidance"""
        
        recommendations = []
        
        # Guidance accuracy recommendations
        guidance_accuracy = analysis.get('guidance_accuracy', {})
        overall_accuracy = guidance_accuracy.get('overall_accuracy', 0)
        
        if overall_accuracy < 0.6:
            recommendations.append("Low guidance accuracy - consider fine-tuning LLM on combat data")
        elif overall_accuracy < 0.8:
            recommendations.append("Moderate guidance accuracy - expand training data diversity")
        
        # Intervention timing recommendations
        timing_analysis = analysis.get('intervention_timing', {})
        optimal_frequency = timing_analysis.get('optimal_intervention_frequency', 10.0)
        current_frequency = timing_analysis.get('current_intervention_rate', 10.0)
        
        if abs(optimal_frequency - current_frequency) > 3.0:
            recommendations.append(f"Adjust intervention frequency to {optimal_frequency:.1f} Hz for optimal effectiveness")
        
        # Knowledge gap recommendations
        knowledge_gaps = analysis.get('tactical_knowledge_gaps', {})
        gap_categories = knowledge_gaps.get('knowledge_gap_categories', {})
        
        for category, gap_count in gap_categories.items():
            if gap_count >= 3:
                recommendations.append(f"Address {category} knowledge gaps through specialized training data")
        
        # Response quality recommendations
        response_quality = analysis.get('response_quality', {})
        overall_quality = response_quality.get('overall_response_quality', 0)
        
        if overall_quality < 0.7:
            recommendations.append("Improve response quality - enhance prompt engineering")
        
        # Shaping effectiveness recommendations
        shaping_impact = analysis.get('reward_shaping_effectiveness', {})
        shaping_correlation = shaping_impact.get('shaping_correlation_with_success', 0)
        
        if abs(shaping_correlation) < 0.3:
            recommendations.append("Low shaping correlation - review reward shaping strategy")
        
        return recommendations
    
    def export_analysis_report(self, analysis_results: Dict[str, Any], 
                             filepath: str = None) -> str:
        """Export comprehensive LLM analysis report"""
        
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"analysis/reports/llm_effectiveness_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare export data
        export_data = {
            'analysis_summary': analysis_results,
            'interaction_statistics': {
                'total_interactions': len(self.interaction_history),
                'analysis_window': self.analysis_window,
                'latest_interactions': len(self.interaction_history)
            },
            'effectiveness_metrics': self.effectiveness_metrics,
            'intervention_patterns': self.intervention_patterns,
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"[LLM ANALYZER] Analysis report exported: {filepath}")
        return filepath
    
    def create_effectiveness_visualization(self, save_path: str = "analysis/plots/"):
        """Create visualizations of LLM effectiveness"""
        
        if not VISUALIZATION_AVAILABLE:
            print("[LLM ANALYZER] Visualization dependencies not available")
            return None
        
        if not self.interaction_history:
            print("[LLM ANALYZER] No data for visualization")
            return None
        
        os.makedirs(save_path, exist_ok=True)
        
        # Prepare data
        timestamps = [record.timestamp for record in self.interaction_history]
        effectiveness_scores = [self._calculate_intervention_effectiveness(record) 
                              for record in self.interaction_history]
        shaping_deltas = [record.shaping_delta for record in self.interaction_history]
        response_times = [record.response_time for record in self.interaction_history]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Effectiveness over time
        axes[0, 0].plot(effectiveness_scores)
        axes[0, 0].set_title('LLM Effectiveness Over Time')
        axes[0, 0].set_ylabel('Effectiveness Score')
        axes[0, 0].set_xlabel('Interaction Number')
        
        # Shaping delta distribution
        axes[0, 1].hist(shaping_deltas, bins=20, alpha=0.7)
        axes[0, 1].set_title('Reward Shaping Distribution')
        axes[0, 1].set_xlabel('Shaping Delta')
        axes[0, 1].set_ylabel('Frequency')
        
        # Response time distribution
        axes[1, 0].hist(response_times, bins=20, alpha=0.7)
        axes[1, 0].set_title('LLM Response Time Distribution')
        axes[1, 0].set_xlabel('Response Time (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Effectiveness vs shaping correlation
        axes[1, 1].scatter(shaping_deltas, effectiveness_scores, alpha=0.6)
        axes[1, 1].set_title('Effectiveness vs Shaping Delta')
        axes[1, 1].set_xlabel('Shaping Delta')
        axes[1, 1].set_ylabel('Effectiveness Score')
        
        plt.tight_layout()
        
        plot_path = os.path.join(save_path, f"llm_effectiveness_{int(time.time())}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[LLM ANALYZER] Effectiveness visualization saved: {plot_path}")
        return plot_path
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary"""
        
        if not self.interaction_history:
            return {'status': 'no_data'}
        
        # Analyze recent performance
        recent_interactions = list(self.interaction_history)[-100:] if len(self.interaction_history) >= 100 else list(self.interaction_history)
        
        recent_effectiveness = np.mean([self._calculate_intervention_effectiveness(record) 
                                      for record in recent_interactions])
        
        recent_success_rate = np.mean([record.success_outcome for record in recent_interactions])
        
        # Calculate trends
        if len(self.interaction_history) >= 200:
            early_effectiveness = np.mean([self._calculate_intervention_effectiveness(record) 
                                         for record in list(self.interaction_history)[:100]])
            effectiveness_trend = recent_effectiveness - early_effectiveness
        else:
            effectiveness_trend = 0.0
        
        return {
            'total_interactions': len(self.interaction_history),
            'recent_effectiveness': recent_effectiveness,
            'recent_success_rate': recent_success_rate,
            'effectiveness_trend': effectiveness_trend,
            'optimal_frequency': self._find_optimal_intervention_frequency(),
            'knowledge_gaps_identified': len(self._identify_knowledge_gaps([])['tactical_situations_with_poor_guidance']),
            'overall_llm_performance': 'excellent' if recent_effectiveness > 0.8 else 
                                     'good' if recent_effectiveness > 0.6 else
                                     'fair' if recent_effectiveness > 0.4 else 'poor'
        }





if __name__ == "__main__":
    print("LLM Effectiveness Analyzer for Combat Training Optimization")
    
    # Create analyzer
    analyzer = LLMEffectivenessAnalyzer(verbose=True)
    
    print("Analyzer ready for LLM performance analysis")
    print("Use record_llm_interaction() to log interactions")
    print("Use analyze_llm_impact() for comprehensive analysis")
