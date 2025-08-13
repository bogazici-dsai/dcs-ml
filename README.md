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

## 🚀 Quick Start Guide

### Prerequisites
Before you begin, ensure you have:
- Python 3.8+ installed
- Git for cloning the repository
- Ollama installed for LLM support (see installation below)
- Harfang3D environment set up

### Step 1: Clone and Setup Repository
```bash
# Clone the repository
git clone https://github.com/bogazici-dsai/dcs-ml.git
cd dcs-ml

# Switch to the enhanced branch
git checkout harfang-rl-llm
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment (Linux/Mac)
source venv/bin/activate
# OR on Windows
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Install Ollama and LLM Model

#### Option A: macOS (using Homebrew)
```bash
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve

# In a new terminal, pull the LLM model
ollama pull llama3.1:8b
```

#### Option B: Linux
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# In a new terminal, pull the LLM model
ollama pull llama3.1:8b
```

#### Option C: Windows
```bash
# Download and install Ollama from https://ollama.com/download/windows
# Then run:
ollama serve
ollama pull llama3.1:8b
```

#### Alternative LLM Models
```bash
# For faster but less capable models:
ollama pull llama3.1:8b    # Recommended (4GB)
ollama pull mistral:7b     # Alternative (4GB) 
ollama pull llama3.2:3b    # Lighter option (2GB)

# For more capable models (if you have >16GB RAM):
ollama pull llama3.1:70b   # High performance (40GB)
```

### Step 4: Setup Harfang3D Environment
```bash
# Follow official Harfang3D installation instructions
# Ensure Harfang3D is properly installed and accessible
```

### Step 5: Quick Test Run
```bash
# Activate environment using the provided script
source activate_env.sh

# Run a quick test with 3 episodes
python3 harfang_rl_llm.py --episodes 3 --max_steps 500 --verbose

# Check the generated log files in data/harfang_tactical_logs/
```

### Step 6: Full Training Run
```bash
# Run full training session with enhanced logging
python3 harfang_rl_llm.py \
    --episodes 50 \
    --max_steps 2000 \
    --detailed_logging \
    --tactical_analysis \
    --llm_model llama3.1:8b
```

## 📋 Detailed Usage

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

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. LangChain Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'langchain_core'
# Solution: Ensure virtual environment is activated
source activate_env.sh
pip install langchain-core langchain-ollama
```

#### 2. Ollama Connection Issues
```bash
# Error: Connection refused to Ollama
# Solution: Start Ollama service
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

#### 3. Harfang Environment Not Found
```bash
# Error: Cannot import HarfangEnv
# Solution: Check Harfang3D installation and environment variables
# Ensure Harfang3D is in your Python path
```

#### 4. CUDA/GPU Issues
```bash
# If you have GPU issues, run in CPU mode
export CUDA_VISIBLE_DEVICES=""
python3 harfang_rl_llm.py --episodes 5
```

### Environment Variables
```bash
# Optional: Set environment variables for better performance
export OLLAMA_HOST=localhost:11434
export HARFANG_DATA_PATH=/path/to/harfang/data
export PYTHONPATH=$PYTHONPATH:/path/to/harfang
```

### Verification Steps
```bash
# 1. Test Python imports
python3 -c "from langchain_ollama import ChatOllama; print('✅ LangChain OK')"
python3 -c "from env.hirl.environments.HarfangEnv_GYM_Enhanced import HarfangEnhancedEnv; print('✅ Enhanced Env OK')"

# 2. Test Ollama connection
python3 -c "from langchain_ollama import ChatOllama; chat = ChatOllama(model='llama3.1:8b'); print('✅ Ollama OK')"

# 3. Run system test
python3 test_enhanced_system.py
```

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

## 📊 Understanding the Output

### What to Expect During a Run
When you run the system, you'll see output like this:
```
============================================================
ENHANCED HARFANG RL-LLM TACTICAL INTEGRATION
============================================================
LLM Model: llama3.1:8b (Rate: 10.0 Hz)
Environment: enhanced | Episodes: 10 | Max Steps: 1500
Agent: random | Continuous Feedback: True

[EPISODE 1/10] Starting...
  Step 500: Reward=45.2, Distance=2500m, Locked=True
[EPISODE 1] VICTORY | Reward: 67.8 | Steps: 1245 | Time: 78.3s
  Lock: 23.4% | Shots: 12 (41.7% acc) | Min Dist: 1250m
```

### Key Metrics to Monitor
- **Victory Rate**: Higher is better (aim for >30% with random agent)
- **Lock Percentage**: Time spent with radar lock (aim for >20%)
- **Shot Accuracy**: Successful hits / total shots (>30% is good)
- **Tactical Efficiency**: Reward per step (>0.5 indicates good performance)

### Generated Files Location
After each run, check the `data/harfang_tactical_logs/` directory:
```
data/harfang_tactical_logs/
├── harfang_enhanced_20240115_143022.csv      # Main tactical data
├── harfang_enhanced_llm_20240115_143022.csv  # LLM interactions
├── harfang_enhanced_events_20240115_143022.csv  # Combat events
├── harfang_enhanced_metrics_20240115_143022.csv # Episode metrics
└── harfang_enhanced_summary_20240115_143022.json # Session summary
```

### Performance Expectations
- **First Run**: May take longer as models load and optimize
- **Typical Episode**: 60-120 seconds depending on max_steps
- **LLM Calls**: ~10 per second (adjustable with --llm_rate_hz)
- **Memory Usage**: ~2-4GB RAM during operation
- **Disk Usage**: ~1-5MB of logs per episode

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

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **Memory**: 8GB+ RAM (16GB recommended for larger models)
- **Storage**: 10GB+ free space for models and logs
- **GPU**: Optional but recommended for LLM acceleration

### Software Dependencies
- Python 3.8+
- Ollama (for LLM inference)
- Harfang3D environment
- Dependencies listed in `requirements.txt`

## Implementation Guide

### Basic Implementation (5 minutes)
```bash
# 1. Quick setup for testing
git clone https://github.com/bogazici-dsai/dcs-ml.git
cd dcs-ml && git checkout harfang-rl-llm
source activate_env.sh

# 2. Install Ollama and model
brew install ollama  # or use appropriate installer for your OS
ollama serve &
ollama pull llama3.1:8b

# 3. Quick test run
python3 harfang_rl_llm.py --episodes 3 --max_steps 500
```

### Production Implementation (15 minutes)
```bash
# 1. Complete setup
git clone https://github.com/bogazici-dsai/dcs-ml.git
cd dcs-ml && git checkout harfang-rl-llm

# 2. Create clean environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Install and configure Ollama
ollama serve &
ollama pull llama3.1:8b
export OLLAMA_HOST=localhost:11434

# 4. Full training run
python3 harfang_rl_llm.py \
    --episodes 50 \
    --max_steps 2000 \
    --detailed_logging \
    --tactical_analysis \
    --csv_dir ./results/
```

### Custom Configuration
```bash
# Example: High-performance configuration
python3 harfang_rl_llm.py \
    --env_version enhanced \
    --llm_model llama3.1:8b \
    --llm_rate_hz 15.0 \
    --episodes 100 \
    --max_steps 3000 \
    --detailed_logging \
    --tactical_analysis \
    --action_space_evolution \
    --random_reset \
    --csv_dir ./experiment_results/ \
    --log_prefix custom_run

# Example: Fast testing configuration  
python3 harfang_rl_llm.py \
    --episodes 5 \
    --max_steps 200 \
    --llm_rate_hz 5.0 \
    --llm_model llama3.2:3b
```

## Understanding Results

### Performance Metrics
- **Victory Rate**: >30% is good for random agent, >60% for trained agent
- **Shot Accuracy**: >25% indicates effective weapons employment
- **Lock Percentage**: >20% shows good radar management
- **Tactical Efficiency**: >0.5 reward/step indicates tactical competence

### Output Files Analysis
Check `data/harfang_tactical_logs/` after each run:
- **Main CSV**: Complete tactical data for analysis
- **LLM CSV**: LLM interaction logs and performance
- **Events CSV**: Significant combat events
- **Metrics CSV**: Episode-level performance summaries