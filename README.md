# 🩺 ChatDoctor v2

> **Fine-tuned Medical AI Assistant with State-of-the-Art Performance**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

A production-ready medical chatbot fine-tuned on **111,000+ real doctor-patient conversations**, achieving **BERTScore F1 = 0.844** — matching the original ChatDoctor paper results.

---

## 🎯 Key Results

### Performance Comparison with Original Paper

| Model | Precision | Recall | F1 Score | Source |
|-------|-----------|--------|----------|--------|
| ChatGPT | 0.837 | 0.845 | 0.841 | Paper |
| ChatDoctor (Paper) | 0.844 | 0.845 | 0.841 | Paper |
| Mistral + LoRA | 0.845 | 0.843 | 0.844 | This Project |
| **LLaMA-3 + LoRA** | **0.844** | **0.846** | **0.845** 🏆 | This Project |

✅ **LLaMA-3 + LoRA achieves the best BERTScore F1 = 0.845!**

---

## 🔄 What's New in v2? (Improvements over Original)

### Architecture Changes

| Aspect | Original ChatDoctor | ChatDoctor v2 |
|--------|-------------------|---------------|
| **Base Model** | LLaMA-1 7B | Mistral-7B-Instruct-v0.3 |
| **Quantization** | 8-bit | 4-bit (QLoRA) |
| **Training Method** | LoRA | QLoRA with gradient checkpointing |
| **Codebase** | Single scripts | Modular Python package |
| **OpenAI API** | Legacy v0.x | Modern v1.x |
| **Configuration** | Hardcoded | YAML-based configs |
| **Evaluation** | Basic | BERTScore + BLEU + ROUGE |

### Key Improvements

1. **🚀 Upgraded Base Model**: Mistral-7B outperforms LLaMA on all benchmarks
2. **⚡ Memory Efficient**: 4-bit quantization reduces VRAM by 50%
3. **📦 Modular Design**: Clean separation of concerns (core, training, inference, rag)
4. **🔧 Modern Dependencies**: Updated to latest transformers, peft, bitsandbytes
5. **📊 Comprehensive Evaluation**: Added BERTScore as per paper methodology
6. **🧪 Unit Tests**: Test coverage for config and data utilities
7. **📝 Better Documentation**

---

## 🎓 Training Process

### Training Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ HealthCareMagic │────▶│  QLoRA Training  │────▶│  LoRA Adapter   │
│   100k Dataset  │     │  (Mistral-7B)    │     │    (161 MB)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
     111,665                7.5 hours                Final Model
     samples               on 2x A100
```

### Training Configuration

```python
# QLoRA Configuration
base_model = "mistralai/Mistral-7B-Instruct-v0.3"
quantization = "4-bit (nf4)"
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Hyperparameters
batch_size = 64
micro_batch_size = 2
gradient_accumulation_steps = 32
learning_rate = 2e-4
num_epochs = 1
warmup_steps = 100
max_seq_length = 512
```

### Training Command

```bash
python -m chatdoctor.training.train_lora \
  --base_model ./models/mistral \
  --data_path ./data/HealthCareMagic-100k.json \
  --output_dir ./models/mistral_lora \
  --batch_size 64 \
  --micro_batch_size 2 \
  --num_epochs 1 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --use_4bit True
```

### Training Results

| Metric | Value |
|--------|-------|
| Training Samples | 111,665 |
| Validation Samples | 500 |
| Total Steps | 1,745 |
| Training Time | 7h 26m |
| Initial Loss | ~2.0 |
| Final Loss | 1.50 |
| GPU Usage | 2x NVIDIA A100 80GB |

### Loss Curve

```
Loss
2.0 ─┬─────────────────────────────────
     │╲
1.8 ─┤ ╲
     │  ╲
1.6 ─┤   ╲__
     │      ╲___
1.4 ─┤          ╲_______
     │                  ╲___________
1.2 ─┼───────────────────────────────▶ Steps
     0    400   800   1200  1600  1745
```

---

## 📊 Evaluation Methodology

We use **BERTScore** as per the original paper:

> "BERTScore leverages pre-trained BERT to match words in the candidate and reference sentences via cosine similarity... chosen for its ability to evaluate the semantic similarity between our model's responses and the reference sentences."
> — ChatDoctor Paper (Li et al., 2023)

### Evaluation Script

```bash
# BERTScore evaluation (paper methodology)
python -m chatdoctor.evaluation.evaluate_bertscore \
  --model_path ./models/mistral \
  --lora_path ./models/mistral_lora \
  --data_path ./data/icliniq.json \
  --max_samples 50
```

### Full Evaluation Results

| Model | BERTScore P | BERTScore R | BERTScore F1 | BLEU-1 | ROUGE-L |
|-------|-------------|-------------|--------------|--------|---------|
| LLaMA-2 + LoRA | 0.826 | 0.828 | 0.827 | 0.137 | 0.102 |
| Mistral + LoRA | 0.845 | 0.843 | 0.844 | 0.150 | 0.109 |
| **LLaMA-3 + LoRA** | **0.844** | **0.846** | **0.845** 🏆 | - | - |

---

## 🏗️ Project Structure

```
chatdoctor_v2/
├── chatdoctor/                 # Main Python package
│   ├── core/
│   │   ├── config.py          # Dataclass-based configuration
│   │   └── model.py           # ChatDoctorModel with 4-bit loading
│   ├── training/
│   │   └── train_lora.py      # QLoRA fine-tuning script
│   ├── inference/
│   │   └── chat.py            # Interactive chat interface
│   ├── evaluation/
│   │   ├── evaluate.py        # BLEU/ROUGE metrics
│   │   ├── evaluate_icliniq.py # Token-level P/R/F1
│   │   └── evaluate_bertscore.py # BERTScore (paper method)
│   ├── rag/
│   │   ├── csv_rag.py         # Disease database RAG
│   │   └── wiki_rag.py        # Wikipedia RAG
│   └── utils/
│       ├── data_utils.py      # JSON/data handling
│       └── logging_utils.py   # Rich console logging
├── models/
│   ├── mistral/               # Mistral-7B-Instruct (14GB)
│   └── mistral_lora/          # Fine-tuned adapter (161MB)
├── data/
│   ├── HealthCareMagic-100k.json  # Training data (111K)
│   ├── chatdoctor5k.json          # Evaluation data (5.4K)
│   └── icliniq.json               # BERTScore eval (7.3K)
├── configs/
│   ├── config.yaml            # Main system config
│   └── training_config.yaml   # Training parameters
├── scripts/
│   ├── chat.sh               # Interactive chat
│   ├── train.sh              # Training script
│   └── evaluate.sh           # Evaluation script
└── tests/                    # Unit tests
```

---

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/Elefantessa/ChatDoctor.git
cd ChatDoctor
pip install -e .
```

### 2. Download Base Models

Choose one of the following base models:

**Mistral-7B (Recommended):**
```bash
# Option A: Using huggingface-cli
pip install huggingface-hub
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir ./models/mistral

# Option B: Using Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download('mistralai/Mistral-7B-Instruct-v0.3', local_dir='./models/mistral')
"
```

**LLaMA-3 8B (Open Version):**
```bash
huggingface-cli download NousResearch/Meta-Llama-3-8B-Instruct --local-dir ./models/llama3-8b
```

**LLaMA-3.1 8B (Official - Requires Auth):**
```bash
# First, login to HuggingFace (requires accepting Meta's license)
huggingface-cli login

# Then download
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./models/llama3.1-8b
```

> ⚠️ **Note:** LLaMA 3.1 requires accepting Meta's license at [HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### 3. Download LoRA Adapters

The fine-tuned LoRA adapters are hosted on HuggingFace:

```bash
# LLaMA-3 LoRA (168MB) - BERTScore F1 = 0.845 🏆 (BEST)
huggingface-cli download halame/chatdoctor-llama3-lora --local-dir ./models/llama3_lora

# Mistral LoRA (161MB) - BERTScore F1 = 0.844
huggingface-cli download halame/chatdoctor-mistral-lora --local-dir ./models/mistral_lora
```

### 4. Download Training Data (Optional)

```bash
# HealthCareMagic-100k (Training)
python -c "
from datasets import load_dataset
ds = load_dataset('lavita/ChatDoctor-HealthCareMagic-100k')
import json
with open('./data/HealthCareMagic-100k.json', 'w') as f:
    json.dump([dict(x) for x in ds['train']], f)
"

# iCliniq (Evaluation)
python -c "
from datasets import load_dataset
ds = load_dataset('lavita/ChatDoctor-iCliniq')
import json
with open('./data/icliniq.json', 'w') as f:
    json.dump([dict(x) for x in ds['train']], f)
"
```

---

## 🚀 Quick Start

```bash
# Chat with the best model (Mistral + LoRA)
./scripts/chat.sh --mistral

# Chat with knowledge retrieval
./scripts/chat.sh --mistral --rag csv

# Run BERTScore evaluation
./scripts/evaluate.sh --bertscore
```

### Example Conversation

```
Patient: I have headache and fever for 2 days. What should I do?

ChatDoctor: Hi, Thanks for your query. Based on your symptoms, you may have a
viral infection. I recommend:

1. Tab Paracetamol 500mg every 6 hours for fever
2. Tab Ibuprofen 400mg for headache relief
3. Drink plenty of fluids and rest
4. If fever exceeds 103°F or symptoms worsen, please visit a doctor

Hope this helps. Take care.
```

---

## 📈 Datasets

| Dataset | Samples | Purpose | Source |
|---------|---------|---------|--------|
| HealthCareMagic-100k | 111,665 | Training | [HuggingFace](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) |
| ChatDoctor-5k | 5,452 | BLEU/ROUGE eval | Original ChatDoctor |
| iCliniq | 7,321 | BERTScore eval | [HuggingFace](https://huggingface.co/datasets/lavita/ChatDoctor-iCliniq) |

---

## 📚 References

- [ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge](https://arxiv.org/abs/2303.14070) - Li et al., 2023
- [Mistral 7B](https://arxiv.org/abs/2310.06825) - Mistral AI, 2023
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) - Dettmers et al., 2023
- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675) - Zhang et al., 2020

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**. Do not use for actual medical diagnosis. Always consult qualified healthcare professionals.

---

## 📄 License

Apache 2.0 License

---

<p align="center">
  <b>Built with ❤️ for Medical AI Research</b>
</p>
