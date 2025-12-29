# ChatDoctor API Documentation

## Core Module

### ChatDoctorModel

The main model class for medical question answering.

```python
from chatdoctor.core.model import ChatDoctorModel

# Load model with LoRA adapter
model = ChatDoctorModel.from_pretrained(
    model_path="./models/mistral",
    lora_path="./models/mistral_lora",
    load_in_4bit=True,
    device_map="auto"
)

# Generate response
response = model.chat("I have headache and fever")
print(response)

# Generate with custom parameters
response = model.generate(
    "What are symptoms of diabetes?",
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9
)
```

### Configuration

```python
from chatdoctor.core.config import ChatDoctorConfig

# Load from YAML
config = ChatDoctorConfig.from_yaml("configs/config.yaml")

# Access settings
print(config.model.base_model_path)
print(config.training.learning_rate)
```

---

## Training Module

### train_lora.py

Fine-tune models using QLoRA.

```bash
python -m chatdoctor.training.train_lora \
  --base_model ./models/mistral \
  --data_path ./data/HealthCareMagic-100k.json \
  --output_dir ./models/my_lora \
  --batch_size 64 \
  --micro_batch_size 2 \
  --num_epochs 1 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --use_4bit True
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | - | Path to base model |
| `data_path` | - | Path to training JSON |
| `output_dir` | - | Where to save adapter |
| `batch_size` | 64 | Total batch size |
| `micro_batch_size` | 2 | Per-device batch size |
| `num_epochs` | 1 | Number of epochs |
| `learning_rate` | 2e-4 | Learning rate |
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA alpha |
| `lora_dropout` | 0.05 | LoRA dropout |
| `use_4bit` | True | 4-bit quantization |

---

## Evaluation Module

### BERTScore Evaluation

```bash
python -m chatdoctor.evaluation.evaluate_bertscore \
  --model_path ./models/mistral \
  --lora_path ./models/mistral_lora \
  --data_path ./data/icliniq.json \
  --max_samples 50 \
  --output_file results.json
```

### BLEU/ROUGE Evaluation

```bash
python -m chatdoctor.evaluation.evaluate \
  --model_path ./models/mistral \
  --lora_path ./models/mistral_lora \
  --eval_data_path ./data/chatdoctor5k.json \
  --max_samples 50
```

---

## RAG Module

### CSV RAG

```python
from chatdoctor.rag.csv_rag import csv_prompter

response = csv_prompter(
    model.generate,
    model.tokenizer,
    "What causes diabetes?"
)
```

### Wikipedia RAG

```python
from chatdoctor.rag.wiki_rag import wiki_prompter

response = wiki_prompter(
    model.generate,
    model.tokenizer,
    "What is hypertension?"
)
```

---

## CLI Usage

```bash
# Interactive chat
chatdoctor chat --model ./models/mistral --lora ./models/mistral_lora

# With RAG
chatdoctor chat --rag csv

# Training
chatdoctor train --config configs/training_config.yaml
```

---

## Data Format

Training data should be JSON with this structure:

```json
[
  {
    "instruction": "Patient question here",
    "input": "",
    "output": "Doctor's response here"
  }
]
```
