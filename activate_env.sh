#!/bin/bash
# Script to activate the virtual environment for the dcs-ml project

# Change to the project directory
cd "$(dirname "$0")"

# Activate the virtual environment
source venv/bin/activate

echo "Virtual environment activated for dcs-ml project"
echo "Python: $(which python3)"
echo "Pip: $(which pip)"
echo ""
echo "To run your Harfang RL-LLM script:"
echo "   python3 harfang_rl_llm.py --help"
echo ""
echo "To deactivate when done: deactivate"
