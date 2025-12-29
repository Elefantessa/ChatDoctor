# Training Guide

## Quick Start

```bash
# Fine-tune Mistral on HealthCareMagic
./scripts/train.sh
```

## Detailed Training

### 1. Data Preparation

Training data format (JSON):
```json
[
  {
    "instruction": "If you are a doctor, please answer the medical questions based on the patient's description.",
    "input": "I have headache and fever for 2 days. What should I do?",
    "output": "Based on your symptoms, you may have a viral infection. Take paracetamol for fever..."
  }
]
```

### 2. Training Command

```bash
python -m chatdoctor.training.train_lora \
  --base_model ./models/mistral \
  --data_path ./data/HealthCareMagic-100k.json \
  --output_dir ./models/my_lora \
  --batch_size 64 \
  --micro_batch_size 2 \
  --num_epochs 1 \
  --learning_rate 2e-4 \
  --cutoff_len 512 \
  --val_set_size 500 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --use_4bit True
```

### 3. Monitor Training

```bash
# Watch log
tail -f /tmp/training.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### 4. Resume Training

```bash
python -m chatdoctor.training.train_lora \
  --base_model ./models/mistral \
  --resume_from_checkpoint ./models/my_lora/checkpoint-500
```

---

## Hardware Requirements

| Model | VRAM Required | Training Time |
|-------|---------------|---------------|
| LLaMA-2 7B | 16GB | ~10 hours |
| Mistral 7B | 16GB | ~7 hours |
| LLaMA-3 8B | 20GB | ~8 hours |

---

## Hyperparameter Tuning

| Parameter | Low | Recommended | High |
|-----------|-----|-------------|------|
| Learning Rate | 1e-5 | 2e-4 | 3e-4 |
| LoRA Rank | 8 | 16 | 64 |
| LoRA Alpha | 16 | 32 | 64 |
| Batch Size | 32 | 64 | 128 |
| Epochs | 1 | 1-3 | 5 |

---

## Troubleshooting

### Out of Memory
- Reduce `micro_batch_size` to 1
- Enable gradient checkpointing
- Use 4-bit quantization

### Slow Training
- Increase `micro_batch_size` if VRAM allows
- Disable W&B logging: `export WANDB_MODE=disabled`

### NaN Loss
- Reduce learning rate
- Check data for invalid values
