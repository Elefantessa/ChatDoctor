#!/bin/bash
# Quick start script for ChatDoctor
# Creates symlinks to model and data files from the original project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OLD_PROJECT="$PROJECT_ROOT/../ChatDoctor"

echo "================================================"
echo "ChatDoctor v2 - Setup Script"
echo "================================================"

# Create directories
mkdir -p "$PROJECT_ROOT/models"
mkdir -p "$PROJECT_ROOT/data"

# Link model weights
if [ -d "$OLD_PROJECT/llama-base" ]; then
    if [ ! -e "$PROJECT_ROOT/models/llama-base" ]; then
        ln -s "$OLD_PROJECT/llama-base" "$PROJECT_ROOT/models/llama-base"
        echo "✓ Linked llama-base model"
    else
        echo "• llama-base already exists"
    fi
fi

if [ -d "$OLD_PROJECT/lora_models" ]; then
    if [ ! -e "$PROJECT_ROOT/models/lora_weights" ]; then
        ln -s "$OLD_PROJECT/lora_models" "$PROJECT_ROOT/models/lora_weights"
        echo "✓ Linked LoRA weights"
    else
        echo "• lora_weights already exists"
    fi
fi

# Link data files
if [ -f "$OLD_PROJECT/HealthCareMagic-100k.json" ]; then
    if [ ! -e "$PROJECT_ROOT/data/HealthCareMagic-100k.json" ]; then
        ln -s "$OLD_PROJECT/HealthCareMagic-100k.json" "$PROJECT_ROOT/data/HealthCareMagic-100k.json"
        echo "✓ Linked HealthCareMagic-100k.json"
    else
        echo "• HealthCareMagic-100k.json already exists"
    fi
fi

if [ -f "$OLD_PROJECT/chatdoctor5k.json" ]; then
    if [ ! -e "$PROJECT_ROOT/data/chatdoctor5k.json" ]; then
        ln -s "$OLD_PROJECT/chatdoctor5k.json" "$PROJECT_ROOT/data/chatdoctor5k.json"
        echo "✓ Linked chatdoctor5k.json"
    else
        echo "• chatdoctor5k.json already exists"
    fi
fi

if [ -f "$OLD_PROJECT/Autonomous_ChatDoctor_csv/healthcare_disease_dataset.csv" ]; then
    if [ ! -e "$PROJECT_ROOT/data/healthcare_disease_dataset.csv" ]; then
        ln -s "$OLD_PROJECT/Autonomous_ChatDoctor_csv/healthcare_disease_dataset.csv" "$PROJECT_ROOT/data/healthcare_disease_dataset.csv"
        echo "✓ Linked healthcare_disease_dataset.csv"
    else
        echo "• healthcare_disease_dataset.csv already exists"
    fi
fi

echo ""
echo "================================================"
echo "Setup complete!"
echo ""
echo "To install the package:"
echo "  cd $PROJECT_ROOT"
echo "  source ../myenv/bin/activate"
echo "  pip install -e ."
echo ""
echo "To start chatting:"
echo "  chatdoctor chat"
echo ""
echo "To train a model:"
echo "  chatdoctor train --config configs/config.yaml"
echo "================================================"
