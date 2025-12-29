"""
Interactive chat interface for ChatDoctor.
Supports plain chat, CSV-based RAG, and Wikipedia-based RAG modes.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from chatdoctor.core.config import Config, load_config
from chatdoctor.core.model import ChatDoctorModel

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChatInterface:
    """Interactive chat interface for ChatDoctor."""

    def __init__(
        self,
        model: ChatDoctorModel,
        rag_mode: str = "none",
        use_rich: bool = True,
    ):
        """
        Initialize chat interface.

        Args:
            model: Loaded ChatDoctorModel.
            rag_mode: RAG mode (none, csv, wiki).
            use_rich: Use rich console output.
        """
        self.model = model
        self.rag_mode = rag_mode
        self.use_rich = use_rich and RICH_AVAILABLE

        if self.use_rich:
            self.console = Console()

        self.history = []
        self.csv_prompter = None
        self.wiki_prompter = None

        # Load RAG modules if needed
        if rag_mode == "csv":
            self._load_csv_rag()
        elif rag_mode == "wiki":
            self._load_wiki_rag()

    def _load_csv_rag(self):
        """Load CSV RAG module."""
        try:
            from chatdoctor.rag.csv_rag import csv_prompter
            self.csv_prompter = csv_prompter
            logger.info("CSV RAG module loaded")
        except ImportError:
            logger.warning("CSV RAG module not available")

    def _load_wiki_rag(self):
        """Load Wikipedia RAG module."""
        try:
            from chatdoctor.rag.wiki_rag import wiki_prompter
            self.wiki_prompter = wiki_prompter
            logger.info("Wikipedia RAG module loaded")
        except ImportError:
            logger.warning("Wikipedia RAG module not available")

    def print_welcome(self):
        """Print welcome message."""
        welcome = (
            "🩺 ChatDoctor - Medical AI Assistant\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Ask medical questions and get helpful responses.\n"
            "Type 'exit', 'quit', or 'q' to end the session.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        if self.use_rich:
            self.console.print(Panel(welcome, title="Welcome", border_style="blue"))
        else:
            print(welcome)

        mode_label = {
            "none": "Standard Chat",
            "csv": "Autonomous Mode (Disease Database)",
            "wiki": "Autonomous Mode (Wikipedia)",
        }
        mode_str = f"Mode: {mode_label.get(self.rag_mode, 'Unknown')}"

        if self.use_rich:
            self.console.print(f"[cyan]{mode_str}[/cyan]\n")
        else:
            print(f"{mode_str}\n")

    def get_response(self, user_input: str) -> str:
        """
        Get response for user input.

        Args:
            user_input: User's message.

        Returns:
            Model's response.
        """
        if self.rag_mode == "csv" and self.csv_prompter:
            return self.csv_prompter(
                self.model.model.generate,
                self.model.tokenizer,
                user_input,
            )
        elif self.rag_mode == "wiki" and self.wiki_prompter:
            return self.wiki_prompter(
                self.model.model.generate,
                self.model.tokenizer,
                user_input,
            )
        else:
            return self.model.chat(user_input)

    def run(self):
        """Run interactive chat loop."""
        self.print_welcome()

        while True:
            try:
                # Get user input
                if self.use_rich:
                    user_input = self.console.input("[bold green]Patient:[/bold green] ")
                else:
                    user_input = input("Patient: ")

                user_input = user_input.strip()

                # Check for exit commands
                if user_input.lower() in ["exit", "quit", "q"]:
                    if self.use_rich:
                        self.console.print("\n[yellow]Thank you for using ChatDoctor! Stay healthy! 🌟[/yellow]\n")
                    else:
                        print("\nThank you for using ChatDoctor! Stay healthy!\n")
                    break

                # Skip empty input
                if not user_input:
                    continue

                # Get and display response
                if self.use_rich:
                    with self.console.status("[cyan]Thinking...[/cyan]"):
                        response = self.get_response(user_input)
                    self.console.print(f"\n[bold blue]ChatDoctor:[/bold blue] {response}\n")
                else:
                    print("\nChatDoctor: ", end="", flush=True)
                    response = self.get_response(user_input)
                    print(response)
                    print()

                # Store in history
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": response})

            except KeyboardInterrupt:
                if self.use_rich:
                    self.console.print("\n\n[yellow]Session ended. Goodbye![/yellow]\n")
                else:
                    print("\n\nSession ended. Goodbye!\n")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                if self.use_rich:
                    self.console.print(f"[red]Error: {e}[/red]")
                else:
                    print(f"Error: {e}")


def main(
    model_path: Optional[str] = None,
    lora_path: Optional[str] = None,
    config_file: Optional[str] = None,
    rag_mode: str = "none",
    load_in_4bit: bool = True,
    no_rich: bool = False,
):
    """
    Run interactive ChatDoctor chat.

    Args:
        model_path: Path to base model.
        lora_path: Path to LoRA adapter.
        config_file: Path to config file.
        rag_mode: RAG mode (none, csv, wiki).
        load_in_4bit: Use 4-bit quantization.
        no_rich: Disable rich console output.
    """
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    # Load config
    if config_file:
        config = load_config(config_file)
        model_path = model_path or config.model.base_model
        lora_path = lora_path or config.model.lora_adapter
        load_in_4bit = config.model.load_in_4bit
        rag_mode = config.rag.mode

    # Default model path
    if not model_path:
        model_path = "./models/llama-base"

    # Load model
    logger.info(f"Loading model from: {model_path}")
    if lora_path:
        logger.info(f"Loading LoRA adapter from: {lora_path}")

    model = ChatDoctorModel.from_pretrained(
        model_path=model_path,
        lora_path=lora_path,
        load_in_4bit=load_in_4bit,
    )

    # Create and run chat interface
    chat = ChatInterface(
        model=model,
        rag_mode=rag_mode,
        use_rich=not no_rich,
    )
    chat.run()


if __name__ == "__main__":
    import fire
    fire.Fire(main)
