#!/usr/bin/env python3
"""Test script for ChatDoctor system."""

import sys
import torch

def main():
    print("=" * 50)
    print("ChatDoctor System Test")
    print("=" * 50)
    print()

    # Check CUDA
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load model
    print("Loading ChatDoctor model...")
    from chatdoctor.core.model import ChatDoctorModel

    model = ChatDoctorModel.from_pretrained(
        model_path="./models/llama-base",
        load_in_4bit=True,
        device_map="auto"
    )
    print(f"Model loaded on: {model.device}")
    print()

    # Test basic chat
    print("Testing chat...")
    print("-" * 40)

    test_questions = [
        "What is diabetes?",
        "I have a headache, what should I do?",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\nQ{i}: {question}")
        try:
            response = model.chat(question)
            print(f"A{i}: {response[:300]}..." if len(response) > 300 else f"A{i}: {response}")
        except Exception as e:
            print(f"Error: {e}")

    print()
    print("=" * 50)
    print("TEST COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
