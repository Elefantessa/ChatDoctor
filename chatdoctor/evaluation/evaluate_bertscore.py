"""
ChatDoctor iCliniq Evaluation with BERTScore
As per the original ChatDoctor paper methodology.
"""

import json
from tqdm import tqdm
from bert_score import score as bert_score


def evaluate_with_bertscore(
    model,
    data: list,
    reference_key: str = "answer_icliniq",
    max_samples: int = 50,
):
    """
    Evaluate using BERTScore (semantic similarity).
    """
    references = []
    hypotheses = []

    samples = data[:max_samples]

    print("Generating responses...")
    for sample in tqdm(samples, desc="Generating"):
        question = sample.get("input", "")
        reference = sample.get(reference_key, "")

        if not question or not reference:
            continue

        try:
            hypothesis = model.chat(question)
            references.append(reference)
            hypotheses.append(hypothesis)
        except Exception as e:
            print(f"Error: {e}")
            continue

    print(f"\nComputing BERTScore for {len(references)} samples...")
    P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=True)

    return {
        "samples": len(references),
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
        "precision_std": P.std().item(),
        "recall_std": R.std().item(),
        "f1_std": F1.std().item(),
    }


def main(
    model_path: str = "./models/llama-base",
    lora_path: str = None,
    data_path: str = "./data/icliniq.json",
    max_samples: int = 50,
    reference_key: str = "answer_icliniq",
    output_file: str = None,
):
    """Run BERTScore evaluation."""
    print("=" * 50)
    print("iCliniq Evaluation (BERTScore)")
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
    results = evaluate_with_bertscore(
        model, data,
        reference_key=reference_key,
        max_samples=max_samples
    )

    # Print results
    print("\n" + "=" * 50)
    print("BERTSCORE RESULTS")
    print("=" * 50)
    print(f"Samples: {results['samples']}")
    print(f"Precision: {results['precision']:.4f} ± {results['precision_std']:.4f}")
    print(f"Recall:    {results['recall']:.4f} ± {results['recall_std']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f} ± {results['f1_std']:.4f}")
    print("=" * 50)

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to: {output_file}")

    return results


if __name__ == "__main__":
    import fire
    fire.Fire(main)
