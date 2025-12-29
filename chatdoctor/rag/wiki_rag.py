"""
Wikipedia-based RAG (Retrieval-Augmented Generation) for ChatDoctor.
Uses Wikipedia as an external knowledge source for responses.
"""

import os
import re
from typing import List, Callable, Optional

import torch

try:
    from llama_index.readers.wikipedia import WikipediaReader
    WIKI_AVAILABLE = True
except ImportError:
    WIKI_AVAILABLE = False


def get_device(generator) -> torch.device:
    """Infer device from the generator/model."""
    try:
        model = generator.__self__
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_on_device(tokenizer, text: str, device: torch.device) -> dict:
    """Tokenize text and move to device."""
    enc = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def extract_keywords(raw_text: str, max_items: int = 8) -> List[str]:
    """Extract keywords from model output."""
    text = raw_text.strip()
    lt = text.lower()

    # Truncate at common junk markers
    for marker in ("```", "###", "solution", "input:", "output:"):
        idx = lt.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    # Remove "Keywords:" prefix
    text = re.sub(r"(?i)^\s*keywords?\s*:\s*", "", text)

    # Treat numbered lists as separators
    text = re.sub(r"\d+[\).\s]+", "|", text)

    # Split on common separators
    tokens = [t.strip(" -") for t in re.split(r"[,\n|]+", text) if t.strip(" -")]

    # Filter noise
    stop = {"the", "and", "or", "of", "for", "with", "to", "how", "cure", "treatment"}
    cleaned = []
    seen = set()

    for t in tokens:
        low = t.lower()
        if len(low) < 3 or low in stop or low in seen:
            continue
        seen.add(low)
        cleaned.append(t)
        if len(cleaned) >= max_items:
            break

    return cleaned


def divide_wiki_pages(wiki_pages: list, word_limit: int = 250) -> List[str]:
    """Divide Wikipedia pages into chunks."""
    divided_text = []

    for doc_list in wiki_pages:
        text_parts = []
        try:
            for doc in doc_list:
                t = getattr(doc, "text", "")
                if t:
                    text_parts.append(t)
        except Exception:
            continue

        words = " ".join(text_parts).split()
        for i in range(0, len(words), word_limit):
            chunk = " ".join(words[i:i + word_limit])
            divided_text.append(chunk)

    return divided_text


def wiki_prompter(
    generator: Callable,
    tokenizer,
    question: str,
    verbose: bool = False,
) -> str:
    """
    Generate a response using Wikipedia-based knowledge retrieval.

    Args:
        generator: Model's generate function.
        tokenizer: Model's tokenizer.
        question: User's question.
        verbose: Print debug information.

    Returns:
        Generated response grounded in Wikipedia knowledge.
    """
    if not WIKI_AVAILABLE:
        return "Error: Wikipedia reader not available. Install llama-index."

    device = get_device(generator)
    verbose = verbose or os.getenv("CHATDOC_VERBOSE") == "1"

    # Step 1: Extract keywords
    kw_prompt = (
        "A question is provided below. Extract 3-8 concise keywords useful for "
        "looking up answers (medical terms, conditions, treatments).\n"
        "---------------------\n"
        f"{question}\n"
        "---------------------\n"
        "Provide keywords in a comma-separated list.\nKeywords: "
    )

    gen_in = tokenize_on_device(tokenizer, kw_prompt, device)

    with torch.no_grad():
        generated_ids = generator(
            **gen_in,
            max_new_tokens=48,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=False,
            repetition_penalty=1.05,
        )

        in_len = gen_in["input_ids"].size(-1)
        new_ids = generated_ids[:, in_len:]
        raw_kw = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0] if new_ids.numel() else ""

    keywords = extract_keywords(raw_kw)

    if verbose:
        print(f"Keywords: {keywords}")

    # Step 2: Fetch Wikipedia pages
    wiki_docs = []
    for kw in keywords:
        page = kw.strip().title()
        try:
            wiki_result = WikipediaReader().load_data(pages=[page], auto_suggest=True)
            if wiki_result:
                wiki_docs.append(wiki_result)
        except Exception:
            if verbose:
                print(f"No wiki page for: {page}")

    # Step 3: Divide into chunks
    chunks = divide_wiki_pages(wiki_docs, 250)

    # Step 4: Score chunks by keyword presence
    chunk_scores = [0] * len(chunks)
    for i, chunk in enumerate(chunks):
        low = chunk.lower()
        for kw in keywords:
            if kw.lower() in low:
                chunk_scores[i] += 1

    # Sort chunks by score
    sorted_chunks = [item for _, item in sorted(zip(chunk_scores, chunks), reverse=True)]
    sorted_chunks.append("_")  # Sentinel

    answer_list = []

    for i, chunk in enumerate(sorted_chunks):
        if i < 4 and i != len(sorted_chunks) - 1:
            ctx_prompt = (
                "Context information is below.\n"
                "---------------------\n"
                f"{chunk}\n"
                "---------------------\n"
                "Given the context and not prior knowledge, answer the question: "
                f"{question}\n"
                "Response: "
            )
        elif i == len(sorted_chunks) - 1 and len(answer_list) > 1:
            ctx_prompt = (
                f"The original question is: {question}\n"
                "We have provided existing answers:\n"
                "------------\n"
                f"{chr(10).join(answer_list)}\n"
                "------------\n"
                "Provide the single best answer in one short paragraph.\n"
                "Final answer: "
            )
        else:
            continue

        if verbose:
            print(ctx_prompt)

        gen_in2 = tokenize_on_device(tokenizer, ctx_prompt, device)

        with torch.no_grad():
            generated_ids = generator(
                **gen_in2,
                max_new_tokens=220,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=False,
                repetition_penalty=1.05,
            )

            in_len2 = gen_in2["input_ids"].size(-1)
            new_ids2 = generated_ids[:, in_len2:]
            text = tokenizer.batch_decode(new_ids2, skip_special_tokens=True)[0] if new_ids2.numel() else ""

        answer = text.strip()

        if verbose:
            print(f"\nAnswer: {answer}\n")

        if answer:
            answer_list.append(answer)

    return answer_list[-1] if answer_list else ""
