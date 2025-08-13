# utils/enhanced_dataset_logger.py - Comprehensive tactical data logger for RL-LLM training
import csv
import os
import time
import json
import numpy as np
from typing import Dict, Any, List, Optional, Union


class TacticalDataLogger:
    """Enhanced logger for comprehensive tactical flight data and LLM interactions"""
    
    def __init__(self, out_dir: str, filename_prefix: str = "harfang_tactical_data", 
                 create_separate_files: bool = True):
        """
        Initialize tactical data logger
        
        Args:
            out_dir: Output directory for log files
            filename_prefix: Prefix for log filenames
            create_separate_files: Whether to create separate files for different data types
        """
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.prefix = filename_prefix
        self.create_separate_files = create_separate_files
        
        # Timestamp for this logging session
        self.session_timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Main tactical data file
        self.main_path = os.path.join(out_dir, f"{filename_prefix}_{self.session_timestamp}.csv")
        self.main_file = open(self.main_path, mode="w", newline="", encoding="utf-8")
        self.main_writer = None
        
        # Separate specialized files if requested
        self.llm_file = None
        self.llm_writer = None
        self.metrics_file = None
        self.metrics_writer = None
        self.events_file = None
        self.events_writer = None
        
        if create_separate_files:
            self._initialize_separate_files()
        
        # Session metadata
        self.session_stats = {
            'start_time': time.time(),
            'total_steps': 0,
            'total_episodes': 0,
            'shots_fired': 0,
            'successful_shots': 0,
            'locks_achieved': 0,
            'total_lock_time': 0,
            'victories': 0,
            'defeats': 0,
            'llm_calls': 0,
            'llm_overrides': 0
        }
        
        # Column definitions for different data types
        self.main_columns = [
            # Episode/Step identification
            'session_id', 'episode', 'step', 'timestamp', 'elapsed_time',
            
            # Basic RL data
            'state_raw', 'action_pitch', 'action_roll', 'action_yaw', 'action_fire',
            'base_reward', 'shaping_delta', 'final_reward', 'done', 'step_success',
            
            # Aircraft state
            'ally_pos_x', 'ally_pos_y', 'ally_pos_z', 'ally_euler_x', 'ally_euler_y', 'ally_euler_z',
            'enemy_pos_x', 'enemy_pos_y', 'enemy_pos_z', 'enemy_euler_x', 'enemy_euler_y', 'enemy_euler_z',
            'relative_pos_x', 'relative_pos_y', 'relative_pos_z',
            
            # Tactical metrics
            'distance', 'closure_rate', 'aspect_angle', 'target_angle', 'altitude', 'climb_rate',
            'g_force', 'turn_rate', 'energy_state', 'engagement_phase', 'threat_level',
            
            # Weapons and sensors
            'target_locked', 'lock_duration', 'time_since_lock', 'missile_available',
            'enemy_health', 'shots_fired_total', 'shots_on_target',
            
            # Tactical assessment
            'tactical_situation', 'pursuit_geometry', 'missile_zone', 'defensive_urgency',
            'in_firing_envelope', 'optimal_range', 'energy_advantage',
            
            # Performance metrics
            'action_smoothness', 'altitude_violations', 'time_in_envelope',
            'evasive_maneuvers', 'consecutive_locks',
            
            # LLM interaction
            'llm_shaping_delta', 'llm_critique', 'llm_recommendations', 'llm_response_time',
            'llm_call_count', 'llm_rate_limited'
        ]
        
        self.llm_columns = [
            'session_id', 'episode', 'step', 'timestamp',
            'llm_call_id', 'input_features', 'llm_prompt', 'llm_response_raw',
            'parsed_shaping_delta', 'parsed_critique', 'parsed_recommendations',
            'response_time_ms', 'rate_limited', 'error_occurred', 'error_message'
        ]
        
        self.metrics_columns = [
            'session_id', 'episode', 'timestamp',
            'episode_length', 'total_reward', 'base_reward_sum', 'llm_shaping_sum',
            'shots_fired', 'shots_successful', 'shot_accuracy', 'victory_achieved',
            'max_lock_duration', 'total_lock_time', 'lock_percentage',
            'avg_distance', 'min_distance', 'time_in_envelope', 'altitude_violations',
            'avg_threat_level', 'max_g_force', 'tactical_efficiency'
        ]
        
        self.events_columns = [
            'session_id', 'episode', 'step', 'timestamp', 'event_type',
            'event_description', 'event_data', 'tactical_significance'
        ]
        
        # Initialize session ID
        self.session_id = f"session_{self.session_timestamp}"
        
        print(f"[TACTICAL LOGGER] Initialized logging session: {self.session_id}")
        print(f"[TACTICAL LOGGER] Main log file: {self.main_path}")

    def _initialize_separate_files(self):
        """Initialize separate specialized log files"""
        # LLM interactions log
        llm_path = os.path.join(self.out_dir, f"{self.prefix}_llm_{self.session_timestamp}.csv")
        self.llm_file = open(llm_path, mode="w", newline="", encoding="utf-8")
        
        # Episode metrics log
        metrics_path = os.path.join(self.out_dir, f"{self.prefix}_metrics_{self.session_timestamp}.csv")
        self.metrics_file = open(metrics_path, mode="w", newline="", encoding="utf-8")
        
        # Tactical events log
        events_path = os.path.join(self.out_dir, f"{self.prefix}_events_{self.session_timestamp}.csv")
        self.events_file = open(events_path, mode="w", newline="", encoding="utf-8")

    def log_step(self, episode: int, step: int, state: np.ndarray, action: np.ndarray,
                 base_reward: float, shaping_delta: float, done: bool,
                 info: Dict[str, Any], features: Dict[str, Any], 
                 llm_response: Dict[str, Any], llm_time_ms: float = 0.0):
        """Log a single step with comprehensive tactical data"""
        
        self.session_stats['total_steps'] += 1
        
        # Build main row
        row = self._build_main_row(episode, step, state, action, base_reward, 
                                  shaping_delta, done, info, features, 
                                  llm_response, llm_time_ms)
        
        # Write to main file
        if self.main_writer is None:
            self.main_writer = csv.DictWriter(self.main_file, fieldnames=self.main_columns)
            self.main_writer.writeheader()
        
        # Fill missing columns with None
        for col in self.main_columns:
            if col not in row:
                row[col] = None
                
        self.main_writer.writerow(row)
        self.main_file.flush()
        
        # Log to specialized files if enabled
        if self.create_separate_files:
            self._log_llm_interaction(episode, step, features, llm_response, llm_time_ms)
            self._log_tactical_events(episode, step, info, features)

    def _build_main_row(self, episode: int, step: int, state: np.ndarray, action: np.ndarray,
                       base_reward: float, shaping_delta: float, done: bool,
                       info: Dict[str, Any], features: Dict[str, Any],
                       llm_response: Dict[str, Any], llm_time_ms: float) -> Dict[str, Any]:
        """Build the main data row with comprehensive tactical information"""
        
        timestamp = time.time()
        elapsed_time = timestamp - self.session_stats['start_time']
        
        # Extract basic state information
        state_list = state.tolist() if hasattr(state, 'tolist') else list(state)
        action_list = action.tolist() if hasattr(action, 'tolist') else list(action)
        
        # Safe extraction with defaults
        def safe_get(source, key, default=None):
            try:
                return source.get(key, default) if isinstance(source, dict) else default
            except:
                return default
        
        # Build comprehensive row
        row = {
            # Episode/Step identification
            'session_id': self.session_id,
            'episode': episode,
            'step': step,
            'timestamp': timestamp,
            'elapsed_time': elapsed_time,
            
            # Basic RL data
            'state_raw': json.dumps(state_list),
            'action_pitch': action_list[0] if len(action_list) > 0 else 0,
            'action_roll': action_list[1] if len(action_list) > 1 else 0,
            'action_yaw': action_list[2] if len(action_list) > 2 else 0,
            'action_fire': action_list[3] if len(action_list) > 3 else 0,
            'base_reward': base_reward,
            'shaping_delta': shaping_delta,
            'final_reward': base_reward + shaping_delta,
            'done': int(done),
            'step_success': safe_get(info, 'step_success', 0),
            
            # Aircraft positions (denormalized approximations)
            'ally_pos_x': safe_get(features, 'plane_euler', [0, 0, 0])[0] if 'plane_euler' in features else 0,
            'ally_pos_y': safe_get(features, 'altitude', 0),
            'ally_pos_z': safe_get(features, 'plane_euler', [0, 0, 0])[2] if 'plane_euler' in features else 0,
            'ally_euler_x': safe_get(features, 'plane_euler', [0, 0, 0])[0] if 'plane_euler' in features else 0,
            'ally_euler_y': safe_get(features, 'plane_euler', [0, 0, 0])[1] if 'plane_euler' in features else 0,
            'ally_euler_z': safe_get(features, 'plane_euler', [0, 0, 0])[2] if 'plane_euler' in features else 0,
            'enemy_euler_x': safe_get(features, 'enemy_euler', [0, 0, 0])[0] if 'enemy_euler' in features else 0,
            'enemy_euler_y': safe_get(features, 'enemy_euler', [0, 0, 0])[1] if 'enemy_euler' in features else 0,
            'enemy_euler_z': safe_get(features, 'enemy_euler', [0, 0, 0])[2] if 'enemy_euler' in features else 0,
            
            # Tactical metrics
            'distance': safe_get(features, 'distance', 0),
            'closure_rate': safe_get(features, 'closure_rate', 0),
            'aspect_angle': safe_get(features, 'aspect_angle', 0),
            'target_angle': safe_get(features, 'target_angle', 0),
            'altitude': safe_get(features, 'altitude', 0),
            'climb_rate': safe_get(features, 'climb_rate', 0),
            'g_force': safe_get(features, 'g_force', 0),
            'turn_rate': safe_get(features, 'turn_rate', 0),
            'energy_state': safe_get(features, 'energy_state', 'UNKNOWN'),
            'engagement_phase': safe_get(features, 'engagement_phase', 'UNKNOWN'),
            'threat_level': safe_get(features, 'threat_level', 0),
            
            # Weapons and sensors
            'target_locked': safe_get(features, 'locked', 0),
            'lock_duration': safe_get(features, 'lock_duration', 0),
            'time_since_lock': safe_get(features, 'time_since_lock', 0),
            'missile_available': safe_get(features, 'missile1_state', 0),
            'enemy_health': safe_get(features, 'enemy_health', 1.0),
            'shots_fired_total': safe_get(info, 'shots_fired', 0),
            'shots_on_target': safe_get(info, 'shots_on_target', 0),
            
            # Tactical assessment
            'tactical_situation': safe_get(features, 'tactical_situation', 'UNKNOWN'),
            'pursuit_geometry': safe_get(features, 'pursuit_geometry', 'UNKNOWN'),
            'missile_zone': safe_get(features, 'missile_employment_zone', 'UNKNOWN'),
            'defensive_urgency': safe_get(features, 'defensive_urgency', 'UNKNOWN'),
            'in_firing_envelope': int(safe_get(features, 'in_firing_envelope', False)),
            'optimal_range': int(safe_get(features, 'optimal_range', False)),
            'energy_advantage': int(safe_get(features, 'energy_advantage', False)),
            
            # Performance metrics
            'action_smoothness': safe_get(features, 'action_smoothness', 0),
            'altitude_violations': safe_get(info, 'altitude_violations', 0),
            'time_in_envelope': safe_get(info, 'time_in_envelope', 0),
            'evasive_maneuvers': safe_get(info, 'evasive_maneuvers', 0),
            'consecutive_locks': safe_get(info, 'consecutive_locks', 0),
            
            # LLM interaction
            'llm_shaping_delta': shaping_delta,
            'llm_critique': safe_get(llm_response, 'critique', ''),
            'llm_recommendations': json.dumps(safe_get(llm_response, 'recommendations', {})),
            'llm_response_time': llm_time_ms,
            'llm_call_count': self.session_stats.get('llm_calls', 0),
            'llm_rate_limited': int(safe_get(llm_response, 'rate_limited', False))
        }
        
        return row

    def _log_llm_interaction(self, episode: int, step: int, features: Dict[str, Any],
                           llm_response: Dict[str, Any], llm_time_ms: float):
        """Log detailed LLM interaction data"""
        if not self.llm_file:
            return
            
        self.session_stats['llm_calls'] += 1
        
        if self.llm_writer is None:
            self.llm_writer = csv.DictWriter(self.llm_file, fieldnames=self.llm_columns)
            self.llm_writer.writeheader()
        
        row = {
            'session_id': self.session_id,
            'episode': episode,
            'step': step,
            'timestamp': time.time(),
            'llm_call_id': self.session_stats['llm_calls'],
            'input_features': json.dumps(features),
            'llm_response_raw': json.dumps(llm_response),
            'parsed_shaping_delta': llm_response.get('shaping_delta', 0),
            'parsed_critique': llm_response.get('critique', ''),
            'parsed_recommendations': json.dumps(llm_response.get('recommendations', {})),
            'response_time_ms': llm_time_ms,
            'rate_limited': int(llm_response.get('rate_limited', False)),
            'error_occurred': int(llm_response.get('error_occurred', False)),
            'error_message': llm_response.get('error_message', '')
        }
        
        self.llm_writer.writerow(row)
        self.llm_file.flush()

    def _log_tactical_events(self, episode: int, step: int, info: Dict[str, Any], 
                           features: Dict[str, Any]):
        """Log significant tactical events"""
        if not self.events_file:
            return
        
        events = []
        
        # Detect significant events
        if info.get('step_success', 0) == 1:
            events.append(('SUCCESSFUL_SHOT', 'Missile fired with target lock', {'locked': True}))
        elif info.get('step_success', 0) == -1:
            events.append(('FAILED_SHOT', 'Missile fired without target lock', {'locked': False}))
        
        if features.get('locked', 0) > 0 and features.get('lock_duration', 0) == 1:
            events.append(('TARGET_LOCK_ACQUIRED', 'Initial target lock achieved', {}))
        
        if features.get('threat_level', 0) > 0.8:
            events.append(('HIGH_THREAT', 'High threat situation detected', {'threat': features.get('threat_level')}))
        
        if features.get('in_firing_envelope', False):
            events.append(('FIRING_ENVELOPE', 'Entered optimal firing envelope', {}))
        
        if features.get('energy_state') == 'LOW':
            events.append(('LOW_ENERGY', 'Low energy state detected', {'altitude': features.get('altitude')}))
        
        # Log events
        if self.events_writer is None and events:
            self.events_writer = csv.DictWriter(self.events_file, fieldnames=self.events_columns)
            self.events_writer.writeheader()
        
        for event_type, description, event_data in events:
            row = {
                'session_id': self.session_id,
                'episode': episode,
                'step': step,
                'timestamp': time.time(),
                'event_type': event_type,
                'event_description': description,
                'event_data': json.dumps(event_data),
                'tactical_significance': self._assess_tactical_significance(event_type, features)
            }
            
            self.events_writer.writerow(row)
        
        if events:
            self.events_file.flush()

    def _assess_tactical_significance(self, event_type: str, features: Dict[str, Any]) -> str:
        """Assess the tactical significance of an event"""
        significance_map = {
            'SUCCESSFUL_SHOT': 'HIGH',
            'FAILED_SHOT': 'MEDIUM',
            'TARGET_LOCK_ACQUIRED': 'HIGH',
            'HIGH_THREAT': 'CRITICAL',
            'FIRING_ENVELOPE': 'HIGH',
            'LOW_ENERGY': 'MEDIUM'
        }
        return significance_map.get(event_type, 'LOW')

    def log_episode_metrics(self, episode: int, episode_data: Dict[str, Any]):
        """Log comprehensive episode-level metrics"""
        if not self.create_separate_files or not self.metrics_file:
            return
        
        self.session_stats['total_episodes'] += 1
        
        if self.metrics_writer is None:
            self.metrics_writer = csv.DictWriter(self.metrics_file, fieldnames=self.metrics_columns)
            self.metrics_writer.writeheader()
        
        # Calculate episode metrics
        row = {
            'session_id': self.session_id,
            'episode': episode,
            'timestamp': time.time(),
            'episode_length': episode_data.get('length', 0),
            'total_reward': episode_data.get('total_reward', 0),
            'base_reward_sum': episode_data.get('base_reward_sum', 0),
            'llm_shaping_sum': episode_data.get('llm_shaping_sum', 0),
            'shots_fired': episode_data.get('shots_fired', 0),
            'shots_successful': episode_data.get('shots_successful', 0),
            'shot_accuracy': episode_data.get('shot_accuracy', 0),
            'victory_achieved': int(episode_data.get('victory', False)),
            'max_lock_duration': episode_data.get('max_lock_duration', 0),
            'total_lock_time': episode_data.get('total_lock_time', 0),
            'lock_percentage': episode_data.get('lock_percentage', 0),
            'avg_distance': episode_data.get('avg_distance', 0),
            'min_distance': episode_data.get('min_distance', float('inf')),
            'time_in_envelope': episode_data.get('time_in_envelope', 0),
            'altitude_violations': episode_data.get('altitude_violations', 0),
            'avg_threat_level': episode_data.get('avg_threat_level', 0),
            'max_g_force': episode_data.get('max_g_force', 0),
            'tactical_efficiency': episode_data.get('tactical_efficiency', 0)
        }
        
        self.metrics_writer.writerow(row)
        self.metrics_file.flush()

    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        runtime = time.time() - self.session_stats['start_time']
        
        return {
            'session_id': self.session_id,
            'runtime_seconds': runtime,
            'total_episodes': self.session_stats['total_episodes'],
            'total_steps': self.session_stats['total_steps'],
            'steps_per_second': self.session_stats['total_steps'] / max(runtime, 1),
            'llm_calls': self.session_stats['llm_calls'],
            'llm_call_rate': self.session_stats['llm_calls'] / max(runtime, 1),
            'shots_fired': self.session_stats['shots_fired'],
            'shot_accuracy': self.session_stats['successful_shots'] / max(self.session_stats['shots_fired'], 1),
            'victories': self.session_stats['victories'],
            'defeat_rate': self.session_stats['defeats'] / max(self.session_stats['total_episodes'], 1)
        }

    def close(self):
        """Close all log files and write session summary"""
        try:
            # Write session summary
            summary = self.get_session_summary()
            summary_path = os.path.join(self.out_dir, f"{self.prefix}_summary_{self.session_timestamp}.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Close all files
            self.main_file.close()
            if self.llm_file:
                self.llm_file.close()
            if self.metrics_file:
                self.metrics_file.close()
            if self.events_file:
                self.events_file.close()
                
            print(f"[TACTICAL LOGGER] Session {self.session_id} closed. Summary written to {summary_path}")
            
        except Exception as e:
            print(f"[TACTICAL LOGGER] Error closing files: {e}")


# Backwards compatibility
class CsvStepLogger(TacticalDataLogger):
    """Backwards compatibility wrapper"""
    
    def __init__(self, out_dir: str, filename_prefix: str = "harfang_rl_llm"):
        super().__init__(out_dir, filename_prefix, create_separate_files=False)
    
    def log(self, row: Dict):
        """Backwards compatible log method"""
        # Extract basic info from row
        episode = row.get('episode', 0)
        step = row.get('step', 0)
        state = np.array([row.get(f'state_{i}', 0) for i in range(25)])  # Dummy state
        action = np.array([row.get('action_pitch', 0), row.get('action_roll', 0), 
                          row.get('action_yaw', 0), row.get('action_fire', 0)])
        base_reward = row.get('base_reward', 0)
        shaping_delta = row.get('shaping_delta', 0)
        done = bool(row.get('done', False))
        
        # Create dummy structures for compatibility
        info = {k: v for k, v in row.items() if k.startswith('info_')}
        features = {k: v for k, v in row.items() if k not in ['episode', 'step', 'base_reward', 'shaping_delta', 'done']}
        llm_response = {'critique': row.get('llm_json', '')}
        
        self.log_step(episode, step, state, action, base_reward, shaping_delta, 
                     done, info, features, llm_response)
