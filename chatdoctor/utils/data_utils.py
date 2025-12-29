"""
Data utilities for ChatDoctor.
Handles data loading, preprocessing, and formatting.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union


def load_json(path: Union[str, Path]) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def format_medical_conversation(
    patient_question: str,
    doctor_response: str,
) -> Dict[str, str]:
    """
    Format a medical conversation for training.

    Args:
        patient_question: Patient's question.
        doctor_response: Doctor's response.

    Returns:
        Formatted data point for training.
    """
    return {
        "instruction": (
            "If you are a doctor, please answer the medical question "
            "based on the patient's description."
        ),
        "input": patient_question,
        "output": doctor_response,
    }


def convert_healthcaremagic_format(
    data: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Convert HealthCareMagic data to training format.

    Args:
        data: List of conversations with 'input' and 'output' keys.

    Returns:
        Formatted training data.
    """
    formatted = []

    for item in data:
        formatted.append({
            "instruction": (
                "If you are a doctor, please answer the medical question "
                "based on the patient's description."
            ),
            "input": item.get("input", ""),
            "output": item.get("output", ""),
        })

    return formatted


def chunk_text(text: str, max_words: int = 250) -> List[str]:
    """Split text into chunks of approximately max_words."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)

    return chunks
