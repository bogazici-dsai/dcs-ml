# Enhanced Harfang RL-LLM Tactical Integration

## Overview

This enhanced system integrates reinforcement learning with large language models (LLMs) for comprehensive tactical guidance in air-to-air combat scenarios using the Harfang3D environment. The system provides continuous, real-time tactical analysis and reward shaping to improve RL agent performance in dogfighting scenarios.

## Key Features

### **Comprehensive Tactical Environment** (`HarfangEnv_GYM_Enhanced.py`)
- **Enhanced State Space**: 25-dimensional observation space (vs. 13 in original)
- **Tactical Metrics**: Closure rate, aspect angle, G-force, turn rate, climb rate
- **Energy Management**: Energy state assessment (HIGH/MEDIUM/LOW)
- **Engagement Classification**: BVR, MERGE, WVR, DEFENSIVE, OFFENSIVE phases
- **Threat Assessment**: Real-time threat level calculation (0-1)
- **Missile Tracking**: Distance, time-to-impact, guidance quality
- **Performance Metrics**: Shot accuracy, lock duration, envelope time

### **Elite Tactical Assistant** (`HarfangAssistant_Enhanced.py`)
- **Fighter Pilot Expertise**: Incorporates real air combat doctrine
- **Tactical Situation Assessment**: Comprehensive analysis of engagement geometry
- **Continuous Guidance**: Real-time tactical feedback with rate limiting
- **Advanced Prompting**: Context-aware prompts with tactical priorities
- **Performance Tracking**: Detailed logging of LLM interactions and effectiveness

### **Continuous LLM Feedback** (vs. step-wise in original)
- **Every Step Analysis**: Continuous tactical assessment instead of every 4 steps
- **Rate Limiting**: Intelligent throttling to prevent LLM overload
- **Adaptive Frequency**: Dynamic adjustment based on tactical situation criticality
- **Fallback Handling**: Graceful degradation when LLM unavailable

### **Comprehensive Data Logging** (`enhanced_dataset_logger.py`)
- **Multi-File Logging**: Separate files for tactical data, LLM interactions, events, metrics
- **Tactical Events**: Automatic detection of significant combat events
- **Performance Analytics**: Episode-level metrics and tactical efficiency
- **CSV + JSON**: Multiple formats for different analysis needs

### **Action Space Optimization** (`action_space_optimizer.py`)
- **Effectiveness Analysis**: Track action effectiveness across different scenarios
- **Macro Action Proposals**: LLM-suggested high-level tactical maneuvers
- **Dynamic Boundaries**: Adaptive action space bounds based on performance
- **Usage Pattern Analysis**: Identify underutilized or overused actions

### **Consolidated Implementation**
- **Single Entry Point**: `harfang_rl_llm.py` with comprehensive configuration
- **Modular Design**: Clean separation of concerns with extensible architecture
- **Backwards Compatibility**: Support for multiple environment versions
- **Error Handling**: Robust error handling and graceful degradation

## Architecture

```
Enhanced Harfang RL-LLM System
├── harfang_rl_llm.py                    # Main integration script
├── HarfangAssistant_Enhanced.py         # Tactical LLM assistant
├── env/hirl/environments/
│   └── HarfangEnv_GYM_Enhanced.py      # Enhanced tactical environment
├── utils/
│   └── enhanced_dataset_logger.py       # Comprehensive logging
└── action_space_optimizer.py            # Dynamic action optimization
```

## Usage

### Basic Usage
```bash
python harfang_rl_llm.py --episodes 10 --detailed_logging
```

### Advanced Configuration
```bash
python harfang_rl_llm.py \
    --env_version enhanced \
    --llm_model llama3.1:8b \
    --llm_rate_hz 10.0 \
    --episodes 50 \
    --max_steps 2000 \
    --detailed_logging \
    --tactical_analysis \
    --action_space_evolution \
    --random_reset
```

### Command Line Options

#### LLM Configuration
- `--llm_model`: LLM model (default: `llama3.1:8b`)
- `--llm_rate_hz`: Max LLM calls per second (default: 10.0)
- `--llm_temperature`: Response temperature (default: 0.0)

#### Environment Configuration
- `--env_version`: Environment version (`v1`|`v2`|`enhanced`|`auto`)
- `--max_steps`: Maximum steps per episode (default: 1500)
- `--episodes`: Number of episodes (default: 10)
- `--random_reset`: Use randomized initial positions

#### Logging Configuration
- `--csv_dir`: Log directory (default: `data/harfang_tactical_logs`)
- `--detailed_logging`: Enable comprehensive multi-file logging
- `--log_prefix`: Log file prefix (default: `harfang_enhanced`)

#### Experimental Features
- `--continuous_feedback`: Continuous LLM feedback (default: enabled)
- `--action_space_evolution`: Dynamic action space optimization
- `--tactical_analysis`: Comprehensive tactical analysis (default: enabled)

## [DATA] Data Output

### Main Tactical Data (`harfang_enhanced_YYYYMMDD_HHMMSS.csv`)
- Episode/step identification
- Aircraft states (positions, orientations, speeds)
- Tactical metrics (distance, closure rate, aspect angle, threat level)
- Weapons status (locks, missiles, shots)
- LLM interactions (shaping, critiques, recommendations)
- Performance metrics (G-force, turn rate, climb rate)

### LLM Interactions (`harfang_enhanced_llm_YYYYMMDD_HHMMSS.csv`)
- Detailed LLM prompt/response pairs
- Response timing and rate limiting
- Error handling and fallback activations
- Parsed recommendations and tactical assessments

### Tactical Events (`harfang_enhanced_events_YYYYMMDD_HHMMSS.csv`)
- Significant combat events (shots, locks, threats)
- Event classification and tactical significance
- Context and situational data

### Episode Metrics (`harfang_enhanced_metrics_YYYYMMDD_HHMMSS.csv`)
- Episode-level performance summaries
- Victory/defeat statistics
- Tactical efficiency metrics
- Shot accuracy and engagement statistics

### Session Summary (`harfang_enhanced_summary_YYYYMMDD_HHMMSS.json`)
- Overall session performance
- LLM usage statistics
- System performance metrics

## Tactical Features

### Fighter Pilot Doctrine
- **BVR Tactics**: Beyond Visual Range missile engagement (8-15km+)
- **ACM (Air Combat Maneuvering)**: Close-in dogfighting techniques
- **Energy Management**: Altitude and airspeed advantage ("speed is life, altitude is life insurance")
- **Engagement Geometry**: Proper aspect angles, pursuit curves, and merge geometry
- **Weapons Employment**: ROE compliance, WEZ (Weapon Engagement Zone) management
- **BFM (Basic Fighter Maneuvers)**: Defensive spirals, barrel rolls, Immelmann turns
- **Knife Fight**: Extremely close-range combat (under 1.5km) - guns only, high-G maneuvering

### Enhanced State Information
- **Comprehensive Telemetry**: 25+ tactical parameters
- **Real-time Assessment**: Continuous situation analysis
- **Multi-phase Combat**: BVR, merge, WVR, defensive engagement phases
- **Energy States**: HIGH/MEDIUM/LOW energy classification
- **Threat Levels**: Dynamic threat assessment (0-1 scale)

### LLM Tactical Guidance
- **Context-Aware Prompts**: Situation-specific tactical advice
- **Rule-Based Safety**: Hard-coded tactical rules and safety constraints
- **Performance Feedback**: Action effectiveness analysis
- **Macro Actions**: High-level tactical maneuver suggestions

## [TOOL] System Requirements

### Dependencies
```bash
# Core ML/RL
numpy
gym

# LLM Integration
langchain-ollama
langchain-core

# Harfang Environment
# (follow official Harfang3D installation instructions)
```

### Hardware Recommendations
- **CPU**: Multi-core processor for parallel LLM inference
- **Memory**: 8GB+ RAM for environment simulation
- **GPU**: Optional but recommended for LLM acceleration
- **Storage**: SSD recommended for fast data logging

## [ALERT] Important Notes

### Environment Priority
The system automatically selects the best available environment:
1. **Enhanced** (if available) - Full tactical analysis
2. **V2** (fallback) - Improved rewards
3. **V1** (basic) - Original implementation

### LLM Rate Limiting
- Default: 10 Hz maximum LLM calls
- Automatic rate limiting prevents API overload
- Reuses recent responses when rate limited
- Fallback to zero shaping when LLM unavailable

### Action Space Evolution
- **Experimental feature** - may affect training stability
- Tracks action effectiveness over time
- Suggests boundary adjustments and macro actions
- Requires sufficient data (100+ samples) for reliable analysis

## [CHART] Performance Monitoring

### Key Metrics to Monitor
- **Victory Rate**: Percentage of successful engagements
- **Shot Accuracy**: Ratio of successful to total shots
- **Lock Percentage**: Time spent with target lock
- **Tactical Efficiency**: Reward per step ratio
- **LLM Effectiveness**: Shaping delta impact

### Tactical Indicators
- **Energy Management**: Altitude maintenance and optimization
- **Engagement Geometry**: Proper aspect angles and positioning
- **Weapons Employment**: Appropriate firing discipline
- **Defensive Awareness**: Threat recognition and evasion

## [FUTURE] Future Enhancements

### Planned Features
- **Multi-Agent Support**: Multiple aircraft engagement scenarios
- **Advanced Macro Actions**: Composite tactical maneuvers
- **Adaptive LLM Prompting**: Dynamic prompt optimization
- **Real-time Visualization**: Tactical situation display
- **Performance Benchmarking**: Standardized evaluation metrics

### Extensibility
- **Custom Agents**: Easy integration of trained RL agents
- **Environment Variants**: Support for different combat scenarios
- **LLM Backends**: Support for different language models
- **Analysis Tools**: Post-flight analysis and replay capabilities

## [INTEGRATION] Integration with Existing System

This enhanced system is designed to be **fully compatible** with the existing HIRL framework while providing significant improvements:

### Maintains Compatibility With:
- Existing training scripts (`train_*.py`)
- Agent implementations (TD3, SAC, PPO, HIRL)
- Environment interfaces (Gym-compatible)
- Configuration files (`local_config.yaml`)

### Enhances Existing Features:
- **Richer state information** for better agent training
- **Intelligent reward shaping** via LLM guidance
- **Comprehensive logging** for detailed analysis
- **Tactical realism** based on fighter pilot doctrine

## Support

For questions, issues, or contributions related to the enhanced RL-LLM integration, please refer to the tactical analysis outputs and session logs for debugging information.

---

**Remember**: This system implements real fighter pilot tactics and doctrine. The LLM guidance is based on established air combat principles including energy management, engagement geometry, and weapons employment rules.
