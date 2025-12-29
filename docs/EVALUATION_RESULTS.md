# ChatDoctor Evaluation Results

## BERTScore Evaluation (Paper Methodology)

Using semantic similarity via BERTScore on iCliniq dataset (50 samples).
Reference: `answer_icliniq` (real doctor answers)

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| **Mistral + LoRA** | **0.845** ± 0.018 | **0.843** ± 0.021 | **0.844** ± 0.015 |
| LLaMA + LoRA | 0.826 ± 0.120 | 0.828 ± 0.121 | 0.827 ± 0.120 |

### Comparison with Paper:
| Model | Paper F1 | Our F1 |
|-------|----------|--------|
| ChatDoctor (Paper) | 0.841 | - |
| **Mistral + LoRA** | - | **0.844** ✓ |

---

## Token-level Evaluation

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Mistral + LoRA | 0.269 | 0.271 | 0.254 |
| LLaMA + LoRA | 0.253 | 0.276 | 0.250 |

---

## Training Summary

| Model | Dataset | Time | Final Loss |
|-------|---------|------|------------|
| LLaMA + LoRA | HealthCareMagic-100k | ~10h | 1.58 |
| Mistral + LoRA | HealthCareMagic-100k | 7h26m | 1.50 |

## Recommendation
✅ **Mistral + LoRA** matches/exceeds ChatDoctor paper performance (F1=0.844)
