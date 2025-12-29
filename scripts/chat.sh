#!/bin/bash
# ChatDoctor Interactive Chat
# Usage: ./chat.sh [OPTIONS]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/../myenv"

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
fi

cd "$PROJECT_ROOT"

# Default values
MODEL="./models/llama-base"
LORA=""
RAG="none"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --mistral)
            MODEL="./models/mistral"
            LORA="./models/mistral_lora"
            ;;
        --lora)
            LORA="./models/lora_weights"
            ;;
    esac
done

# Check for --rag
if [[ "$*" == *"--rag csv"* ]]; then
    RAG="csv"
elif [[ "$*" == *"--rag wiki"* ]]; then
    RAG="wiki"
fi

echo "========================================"
echo "Model: $MODEL"
echo "LoRA: ${LORA:-None}"
echo "RAG: $RAG"
echo "========================================"

python -m chatdoctor.inference.chat \
    --model_path "$MODEL" \
    ${LORA:+--lora_path "$LORA"} \
    --rag_mode "$RAG"
