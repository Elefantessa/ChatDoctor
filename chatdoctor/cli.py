"""
ChatDoctor CLI - Main command-line interface.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ChatDoctor - Medical AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  chatdoctor chat                      # Start interactive chat
  chatdoctor chat --rag csv            # Chat with CSV knowledge base
  chatdoctor train --config config.yaml   # Train a model
  chatdoctor version                   # Show version
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument("--model", "-m", help="Path to base model")
    chat_parser.add_argument("--lora", "-l", help="Path to LoRA adapter")
    chat_parser.add_argument("--config", "-c", help="Path to config file")
    chat_parser.add_argument(
        "--rag",
        choices=["none", "csv", "wiki"],
        default="none",
        help="RAG mode (default: none)",
    )
    chat_parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    chat_parser.add_argument("--no-rich", action="store_true", help="Disable rich console output")

    # Train command
    train_parser = subparsers.add_parser("train", help="Fine-tune a model")
    train_parser.add_argument("--config", "-c", help="Path to config file")
    train_parser.add_argument("--model", "-m", help="Path to base model")
    train_parser.add_argument("--data", "-d", help="Path to training data")
    train_parser.add_argument("--output", "-o", help="Output directory")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, help="Learning rate")

    # Version command
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "chat":
        from chatdoctor.inference.chat import main as chat_main
        chat_main(
            model_path=args.model,
            lora_path=args.lora,
            config_file=args.config,
            rag_mode=args.rag,
            load_in_4bit=not args.no_4bit,
            no_rich=args.no_rich,
        )

    elif args.command == "train":
        from chatdoctor.training.train_lora import train
        train(
            base_model=args.model or "",
            data_path=args.data or "",
            output_dir=args.output or "./lora_output",
            num_epochs=args.epochs or 1,
            learning_rate=args.lr or 3e-4,
            config_file=args.config,
        )

    elif args.command == "version":
        from chatdoctor import __version__
        print(f"ChatDoctor v{__version__}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
