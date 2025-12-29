"""
ChatDoctor iCliniq Evaluation Script
Computes Precision, Recall, F1 (token-level) as per the paper methodology.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm


@dataclass
class TokenMetrics:
    """Token-level metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


def tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase and split on non-alphanumeric."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def compute_token_metrics(reference: str, hypothesis: str) -> TokenMetrics:
    """
    Compute token-level Precision, Recall, F1.

    Precision = |tokens_in_both| / |tokens_in_hypothesis|
    Recall = |tokens_in_both| / |tokens_in_reference|
    F1 = 2 * P * R / (P + R)
    """
    ref_tokens = Counter(tokenize(reference))
    hyp_tokens = Counter(tokenize(hypothesis))

    # Count common tokens (min count for each)
    common = sum((ref_tokens & hyp_tokens).values())

    # Total tokens
    total_hyp = sum(hyp_tokens.values())
    total_ref = sum(ref_tokens.values())

    if total_hyp == 0:
        precision = 0.0
    else:
        precision = common / total_hyp

    if total_ref == 0:
        recall = 0.0
    else:
        recall = common / total_ref

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return TokenMetrics(precision=precision, recall=recall, f1=f1)


def evaluate_model_on_icliniq(
    model,
    data: List[dict],
    reference_key: str = "answer_icliniq",
    max_samples: int = 100,
) -> dict:
    """
    Evaluate model on iCliniq dataset.

    Args:
        model: ChatDoctorModel instance.
        data: List of iCliniq samples.
        reference_key: Which answer to use as reference.
        max_samples: Maximum samples to evaluate.

    Returns:
        Dictionary with average metrics.
    """
    precisions = []
    recalls = []
    f1s = []

    samples = data[:max_samples]

    for sample in tqdm(samples, desc="Evaluating"):
        question = sample.get("input", "")
        reference = sample.get(reference_key, "")

        if not question or not reference:
            continue

        try:
            hypothesis = model.chat(question)
        except Exception as e:
            print(f"Error: {e}")
            continue

        metrics = compute_token_metrics(reference, hypothesis)
        precisions.append(metrics.precision)
        recalls.append(metrics.recall)
        f1s.append(metrics.f1)

    n = len(precisions)
    return {
        "samples": n,
        "precision": sum(precisions) / n if n > 0 else 0,
        "recall": sum(recalls) / n if n > 0 else 0,
        "f1": sum(f1s) / n if n > 0 else 0,
    }


def main(
    model_path: str = "./models/llama-base",
    lora_path: str = None,
    data_path: str = "./data/icliniq.json",
    max_samples: int = 50,
    reference_key: str = "answer_icliniq",
    output_file: str = None,
):
    """
    Run iCliniq evaluation.

    Args:
        model_path: Path to base model.
        lora_path: Path to LoRA adapter.
        data_path: Path to iCliniq JSON file.
        max_samples: Number of samples to evaluate.
        reference_key: Reference answer key (answer_icliniq, answer_chatgpt, answer_chatdoctor).
        output_file: Optional output JSON file.
    """
    print("=" * 50)
    print("iCliniq Evaluation (Precision/Recall/F1)")
    print("=" * 50)
    print(f"Model: {model_path}")
    if lora_path:
        print(f"LoRA: {lora_path}")
    print(f"Reference: {reference_key}")
    print(f"Samples: {max_samples}")
    print()

    # Load model
    print("Loading model...")
    from chatdoctor.core.model import ChatDoctorModel
    model = ChatDoctorModel.from_pretrained(
        model_path=model_path,
        lora_path=lora_path,
        load_in_4bit=True,
        device_map="auto",
    )
    print(f"Model loaded on: {model.device}")

    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")

    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate_model_on_icliniq(
        model, data,
        reference_key=reference_key,
        max_samples=max_samples
    )

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Samples: {results['samples']}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print("=" * 50)

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to: {output_file}")

    return results


if __name__ == "__main__":
    import fire
    fire.Fire(main)
