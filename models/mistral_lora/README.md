---
base_model: mistralai/Mistral-7B-Instruct-v0.3
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
- en
tags:
- medical
- chatdoctor
- lora
- qlora
- transformers
datasets:
- lavita/ChatDoctor-HealthCareMagic-100k
---

# ChatDoctor Mistral LoRA

A fine-tuned LoRA adapter for medical question answering, achieving **BERTScore F1 = 0.844** (matching the original ChatDoctor paper).

## Model Details

- **Base Model:** [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- **Training Data:** [HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) (111,665 samples)
- **Training Time:** 7h 26m on 2x A100 80GB
- **Final Loss:** 1.50
- **Method:** QLoRA (4-bit quantization)

## Performance

| Metric | Score |
|--------|-------|
| BERTScore Precision | 0.845 |
| BERTScore Recall | 0.843 |
| BERTScore F1 | **0.844** |
| BLEU-1 | 0.150 |
| ROUGE-L | 0.109 |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "halame/chatdoctor-mistral-lora")
tokenizer = AutoTokenizer.from_pretrained("halame/chatdoctor-mistral-lora")

# Generate
prompt = "I have headache and fever for 2 days. What should I do?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Configuration

- LoRA r: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Learning rate: 2e-4
- Batch size: 64
- Epochs: 1
- Quantization: 4-bit (nf4)

## Citation

```bibtex
@article{li2023chatdoctor,
  title={ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge},
  author={Li, Yunxiang and others},
  journal={arXiv preprint arXiv:2303.14070},
  year={2023}
}
```

## License

Apache 2.0

## Disclaimer

This model is for research purposes only. Do not use for actual medical diagnosis.
