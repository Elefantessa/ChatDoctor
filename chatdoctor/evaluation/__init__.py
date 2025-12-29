"""ChatDoctor evaluation module."""

from .evaluate import (
    evaluate_model,
    EvaluationResult,
    compute_bleu,
    compute_rouge,
)

__all__ = [
    "evaluate_model",
    "EvaluationResult",
    "compute_bleu",
    "compute_rouge",
]
