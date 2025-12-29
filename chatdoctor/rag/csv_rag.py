"""
CSV-based RAG (Retrieval-Augmented Generation) for ChatDoctor.
Uses a disease/symptom database for knowledge-grounded responses.
"""

import os
import re
from collections import Counter
from pathlib import Path
from typing import List, Callable, Optional

import pandas as pd
import torch


# Stopwords for keyword extraction
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "you", "your", "have", "has",
    "are", "was", "were", "from", "about", "into", "over", "does", "any",
    "been", "they", "them", "but", "when", "what", "where", "which", "while",
    "will", "would", "could", "should", "can", "may", "might", "a", "an", "of",
    "on", "in", "it", "is", "to", "my", "me", "do"
}


def get_device(generator) -> torch.device:
    """Infer device from the generator/model."""
    try:
        model = generator.__self__
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_on_device(tokenizer, text: str, device: torch.device) -> dict:
    """Tokenize text and move to device."""
    tokens = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in tokens.items()}


def extract_keywords_simple(question: str, max_keywords: int = 8) -> List[str]:
    """Extract keywords from a question using simple heuristics."""
    tokens = re.findall(r"[a-zA-Z']+", question.lower())
    filtered = [tok for tok in tokens if len(tok) > 2 and tok not in STOPWORDS]

    if not filtered:
        filtered = tokens

    counts = Counter(filtered)
    ordered = [word for word, _ in counts.most_common()]
    return ordered[:max_keywords]


def is_treatment_query(question: str) -> bool:
    """Check if the question is about treatment."""
    q = question.lower()
    markers = [
        "treat", "treatment", "therapy", "medication", "medicine",
        "manage", "management", "antibiotic", "antivir", "dose",
        "dosing", "drug", "cure", "remedy",
    ]
    return any(m in q for m in markers)


def parse_symptoms(cell) -> List[str]:
    """Parse symptoms from a CSV cell."""
    if not isinstance(cell, str):
        return []

    try:
        import ast
        value = ast.literal_eval(cell)
        if isinstance(value, (list, tuple)):
            return [str(x) for x in value]
    except Exception:
        pass

    # Split on newlines or commas
    lines = [ln.strip(" \t•-") for ln in cell.replace("\r", "\n").split("\n")]
    lines = [ln for ln in lines if ln]
    if lines:
        return lines

    return [s.strip() for s in cell.split(',') if s.strip()]


def parse_treatment(cell) -> List[str]:
    """Parse treatment text into short items."""
    if not isinstance(cell, str):
        return []

    lines = [ln.strip(" \t•-") for ln in cell.replace("\r", "\n").split("\n")]
    return [ln for ln in lines if ln][:3]


def format_record(rec: dict) -> str:
    """Format a database record as a string."""
    disease = str(rec.get('disease', '')).strip()

    # Support both 'Symptom' and 'Symptoms' columns
    sym_cell = rec.get('Symptom', rec.get('Symptoms', ''))
    symptoms = parse_symptoms(sym_cell)
    sym_str = '; '.join(symptoms[:8])

    parts = [f"Disease: {disease}"]
    if sym_str:
        parts.append(f"Symptoms: {sym_str}")

    treat_cell = rec.get('Treatment', '')
    treat_items = parse_treatment(treat_cell)
    if treat_items:
        parts.append(f"Treatment: {'; '.join(treat_items)}")

    return " | ".join(parts)


def score_record(rec: dict, keywords: List[str]) -> int:
    """Score a record based on keyword matches."""
    disease = str(rec.get('disease', '')).lower()
    sym_cell = rec.get('Symptom', rec.get('Symptoms', ''))
    symptoms_txt = ' '.join(parse_symptoms(sym_cell)).lower()
    treatment_txt = ' '.join(parse_treatment(rec.get('Treatment', ''))).lower()

    score = 0
    for kw in keywords:
        if kw in disease:
            score += 3
        if kw in symptoms_txt:
            score += 1
        if kw in treatment_txt:
            score += 1

    return score


def clean_response(text: str) -> str:
    """Clean up generated response."""
    text = text.strip()
    text = re.sub(r"^\s*(?:\d+\)|\d+\.)\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]+\s*", "", text, flags=re.MULTILINE)

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return ""

    cleaned = paragraphs[0]
    cleaned = re.sub(r"\s+", " ", cleaned)

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    if len(sentences) > 3:
        cleaned = " ".join(sentences[:3])

    return cleaned


def csv_prompter(
    generator: Callable,
    tokenizer,
    question: str,
    csv_path: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """
    Generate a response using CSV-based knowledge retrieval.

    Args:
        generator: Model's generate function.
        tokenizer: Model's tokenizer.
        question: User's question.
        csv_path: Path to the disease database CSV.
        verbose: Print debug information.

    Returns:
        Generated response grounded in the database.
    """
    device = get_device(generator)
    keywords = extract_keywords_simple(question)

    if verbose:
        print(f"Keywords: {keywords}")

    # Load CSV
    if csv_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        # Try multiple paths
        possible_paths = [
            base_dir / "data" / "healthcare_disease_dataset.csv",
            base_dir / "data" / "disease_symptom.csv",
        ]
        csv_path = next((p for p in possible_paths if p.exists()), None)

    if csv_path is None or not Path(csv_path).exists():
        return "Error: Disease database not found."

    if verbose:
        print(f"CSV path: {csv_path}")

    df = pd.read_csv(csv_path)
    records = df.to_dict('records')

    # Score and sort records
    scored = [(score_record(rec, keywords), rec) for rec in records]
    scored.sort(key=lambda x: x[0], reverse=True)
    top_records = [rec for s, rec in scored[:24] if s > 0] or [rec for s, rec in scored[:24]]

    # Format context
    lines = [format_record(rec) for rec in top_records]

    # Group into chunks
    block_size = 6
    chunks = ["\n".join(lines[i:i+block_size]) for i in range(0, len(lines), block_size)]

    # Score chunks by keyword presence
    chunk_scores = [0] * len(chunks)
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        for kw in keywords:
            if kw in chunk_lower:
                chunk_scores[i] += 1

    # Sort chunks by score
    sorted_chunks = [item for _, item in sorted(zip(chunk_scores, chunks), reverse=True)]
    sorted_chunks.append("_")  # Sentinel for final aggregation

    answer_list = []

    for i, chunk in enumerate(sorted_chunks):
        if i < 4 and i != len(sorted_chunks) - 1:
            # Generate answer from chunk
            if is_treatment_query(question):
                instruction = (
                    "Using the table above, summarize the likely diagnosis context, "
                    "highlight any red-flag signs that warrant urgent care, and outline "
                    "general management principles. Do not invent specific drug names.\n"
                )
            else:
                instruction = (
                    "Act as ChatDoctor. Using the table above, provide ONE concise paragraph that:\n"
                    "1) Mentions up to three likely conditions with brief reasoning.\n"
                    "2) Highlights red-flag symptoms if present.\n"
                    "3) Advises on next diagnostic or medical steps.\n"
                )

            prompt = (
                f"{chunk}\n"
                "---------------------\n"
                f"Patient question: {question}\n"
                f"{instruction}"
                "Answer: "
            )
        elif i == len(sorted_chunks) - 1 and len(answer_list) > 1:
            # Aggregation step
            prompt = (
                f"Patient question: {question}\n"
                "Candidate answers:\n"
                "------------\n"
                f"{chr(10).join(answer_list)}\n"
                "------------\n"
                "Select the single best response, rewrite it clearly, and keep it to one short paragraph.\n"
                "Final answer: "
            )
        else:
            continue

        if verbose:
            print(prompt)

        gen_in = tokenize_on_device(tokenizer, prompt, device)

        with torch.no_grad():
            generated_ids = generator(
                **gen_in,
                max_new_tokens=220,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=False,
                repetition_penalty=1.05,
            )

            in_len = gen_in["input_ids"].size(-1)
            new_ids = generated_ids[:, in_len:]

            if new_ids.numel() == 0:
                text = ""
            else:
                text = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0]

        cleaned = clean_response(text)

        if verbose:
            print(f"\nAnswer: {cleaned}\n")

        if cleaned:
            answer_list.append(cleaned)

    return answer_list[-1] if answer_list else ""
