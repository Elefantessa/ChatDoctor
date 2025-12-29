#!/bin/bash
# Train ChatDoctor with LoRA
# Usage: ./train.sh [--config PATH] [additional args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/../myenv"

# Activate virtual environment
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
fi

# Default config
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/config.yaml}"

# Run training
cd "$PROJECT_ROOT"
python -m chatdoctor.training.train_lora --config_file "$CONFIG" "$@"
