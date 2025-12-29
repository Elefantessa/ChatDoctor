"""
Evaluation script for ChatDoctor models.
Measures BLEU, ROUGE, and other metrics on medical QA datasets.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    num_samples: int = 0
    bleu_1: float = 0.0
    bleu_2: float = 0.0
    bleu_4: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    avg_response_length: float = 0.0
    avg_reference_length: float = 0.0


def compute_bleu(reference: str, hypothesis: str) -> Tuple[float, float, float]:
    """
    Compute BLEU-1, BLEU-2, and BLEU-4 scores.

    Args:
        reference: Reference text.
        hypothesis: Generated text.

    Returns:
        Tuple of (BLEU-1, BLEU-2, BLEU-4).
    """
    if not NLTK_AVAILABLE:
        return 0.0, 0.0, 0.0

    try:
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())

        if len(hyp_tokens) == 0:
            return 0.0, 0.0, 0.0

        smoothie = SmoothingFunction().method1

        bleu_1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu_2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu_4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

        return bleu_1, bleu_2, bleu_4
    except Exception:
        return 0.0, 0.0, 0.0


def compute_rouge(reference: str, hypothesis: str) -> Tuple[float, float, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        reference: Reference text.
        hypothesis: Generated text.

    Returns:
        Tuple of (ROUGE-1, ROUGE-2, ROUGE-L) F1 scores.
    """
    if not ROUGE_AVAILABLE:
        return 0.0, 0.0, 0.0

    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)

        return (
            scores['rouge1'].fmeasure,
            scores['rouge2'].fmeasure,
            scores['rougeL'].fmeasure,
        )
    except Exception:
        return 0.0, 0.0, 0.0


def load_eval_data(data_path: str, max_samples: int = 100) -> List[Dict]:
    """
    Load evaluation data from JSON file.

    Args:
        data_path: Path to JSON file with instruction/input/output format.
        max_samples: Maximum number of samples to load.

    Returns:
        List of evaluation samples.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Shuffle and limit
    import random
    random.seed(42)
    random.shuffle(data)

    return data[:max_samples]


def evaluate_model(
    model,
    eval_data: List[Dict],
    verbose: bool = False,
) -> EvaluationResult:
    """
    Evaluate model on dataset.

    Args:
        model: ChatDoctorModel instance.
        eval_data: List of evaluation samples.
        verbose: Print individual results.

    Returns:
        EvaluationResult with all metrics.
    """
    results = EvaluationResult(num_samples=len(eval_data))

    bleu_1_scores = []
    bleu_2_scores = []
    bleu_4_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    response_lengths = []
    reference_lengths = []

    for sample in tqdm(eval_data, desc="Evaluating"):
        question = sample.get("input", "")
        reference = sample.get("output", "")

        if not question or not reference:
            continue

        # Generate response
        try:
            hypothesis = model.chat(question)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            continue

        # Compute metrics
        b1, b2, b4 = compute_bleu(reference, hypothesis)
        r1, r2, rl = compute_rouge(reference, hypothesis)

        bleu_1_scores.append(b1)
        bleu_2_scores.append(b2)
        bleu_4_scores.append(b4)
        rouge_1_scores.append(r1)
        rouge_2_scores.append(r2)
        rouge_l_scores.append(rl)
        response_lengths.append(len(hypothesis.split()))
        reference_lengths.append(len(reference.split()))

        if verbose:
            print(f"\nQ: {question[:100]}...")
            print(f"Ref: {reference[:100]}...")
            print(f"Gen: {hypothesis[:100]}...")
            print(f"BLEU-4: {b4:.4f}, ROUGE-L: {rl:.4f}")

    # Compute averages
    if bleu_1_scores:
        results.bleu_1 = sum(bleu_1_scores) / len(bleu_1_scores)
        results.bleu_2 = sum(bleu_2_scores) / len(bleu_2_scores)
        results.bleu_4 = sum(bleu_4_scores) / len(bleu_4_scores)
        results.rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores)
        results.rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores)
        results.rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
        results.avg_response_length = sum(response_lengths) / len(response_lengths)
        results.avg_reference_length = sum(reference_lengths) / len(reference_lengths)

    return results


def print_results(results: EvaluationResult):
    """Pretty print evaluation results."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Samples evaluated: {results.num_samples}")
    print()
    print("BLEU Scores:")
    print(f"  BLEU-1: {results.bleu_1:.4f}")
    print(f"  BLEU-2: {results.bleu_2:.4f}")
    print(f"  BLEU-4: {results.bleu_4:.4f}")
    print()
    print("ROUGE Scores:")
    print(f"  ROUGE-1: {results.rouge_1:.4f}")
    print(f"  ROUGE-2: {results.rouge_2:.4f}")
    print(f"  ROUGE-L: {results.rouge_l:.4f}")
    print()
    print("Length Statistics:")
    print(f"  Avg response length: {results.avg_response_length:.1f} words")
    print(f"  Avg reference length: {results.avg_reference_length:.1f} words")
    print("=" * 50)


def main(
    model_path: str = "./models/llama-base",
    lora_path: Optional[str] = None,
    eval_data_path: str = "./data/chatdoctor5k.json",
    max_samples: int = 50,
    load_in_4bit: bool = True,
    output_file: Optional[str] = None,
    verbose: bool = False,
):
    """
    Run evaluation on a ChatDoctor model.

    Args:
        model_path: Path to base model.
        lora_path: Path to LoRA adapter.
        eval_data_path: Path to evaluation dataset.
        max_samples: Number of samples to evaluate.
        load_in_4bit: Use 4-bit quantization.
        output_file: Optional JSON file to save results.
        verbose: Print detailed output.
    """
    logging.basicConfig(level=logging.INFO)

    print("=" * 50)
    print("ChatDoctor Model Evaluation")
    print("=" * 50)
    print(f"Model: {model_path}")
    if lora_path:
        print(f"LoRA: {lora_path}")
    print(f"Eval data: {eval_data_path}")
    print(f"Samples: {max_samples}")
    print()

    # Check dependencies
    if not NLTK_AVAILABLE:
        print("WARNING: NLTK not available. BLEU scores will be 0.")
        print("Install with: pip install nltk")
    if not ROUGE_AVAILABLE:
        print("WARNING: rouge-score not available. ROUGE scores will be 0.")
        print("Install with: pip install rouge-score")
    print()

    # Load model
    print("Loading model...")
    from chatdoctor.core.model import ChatDoctorModel

    model = ChatDoctorModel.from_pretrained(
        model_path=model_path,
        lora_path=lora_path,
        load_in_4bit=load_in_4bit,
    )
    print(f"Model loaded on: {model.device}")

    # Load data
    print(f"Loading evaluation data from {eval_data_path}...")
    eval_data = load_eval_data(eval_data_path, max_samples)
    print(f"Loaded {len(eval_data)} samples")

    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate_model(model, eval_data, verbose=verbose)

    # Print results
    print_results(results)

    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(asdict(results), f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    import fire
    fire.Fire(main)
