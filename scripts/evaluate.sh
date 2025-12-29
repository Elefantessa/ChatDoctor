#!/bin/bash
# ChatDoctor Evaluation Script
# Usage: ./evaluate.sh [--bertscore] [--model PATH] [--lora PATH] [--samples N]
#
# Examples:
#   ./evaluate.sh                           # BLEU/ROUGE on LLaMA
#   ./evaluate.sh --bertscore               # BERTScore on LLaMA
#   ./evaluate.sh --bertscore --model ./models/mistral --lora ./models/mistral_lora

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/../myenv"

# Activate virtual environment
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
fi

cd "$PROJECT_ROOT"

# Default values
MODEL="${MODEL:-./models/llama-base}"
LORA="${LORA:-}"
SAMPLES="${SAMPLES:-50}"
DATA="${DATA:-./data/chatdoctor5k.json}"
USE_BERTSCORE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bertscore)
            USE_BERTSCORE=true
            DATA="${DATA:-./data/icliniq.json}"
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --lora)
            LORA="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --data)
            DATA="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "========================================"
echo "ChatDoctor Evaluation"
echo "========================================"
echo "Model: $MODEL"
echo "LoRA: ${LORA:-None}"
echo "Samples: $SAMPLES"
echo "Data: $DATA"
echo "Method: $([ "$USE_BERTSCORE" = true ] && echo "BERTScore" || echo "BLEU/ROUGE")"
echo ""

if [ "$USE_BERTSCORE" = true ]; then
    # BERTScore evaluation (paper methodology)
    python -m chatdoctor.evaluation.evaluate_bertscore \
        --model_path "$MODEL" \
        ${LORA:+--lora_path "$LORA"} \
        --data_path "$DATA" \
        --max_samples "$SAMPLES"
else
    # BLEU/ROUGE evaluation
    python -m chatdoctor.evaluation.evaluate \
        --model_path "$MODEL" \
        ${LORA:+--lora_path "$LORA"} \
        --eval_data_path "$DATA" \
        --max_samples "$SAMPLES"
fi
