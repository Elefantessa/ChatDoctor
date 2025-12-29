"""
LoRA/QLoRA Fine-tuning for ChatDoctor.
Modern implementation using PEFT and bitsandbytes.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import fire
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from chatdoctor.core.config import Config, load_config

logger = logging.getLogger(__name__)


# Prompt templates
PROMPT_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

PROMPT_WITHOUT_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


def generate_prompt(data_point: dict, include_output: bool = True) -> str:
    """Generate training prompt from data point."""
    if data_point.get("input"):
        template = PROMPT_WITH_INPUT
    else:
        template = PROMPT_WITHOUT_INPUT

    output = data_point.get("output", "") if include_output else ""

    return template.format(
        instruction=data_point.get("instruction", ""),
        input=data_point.get("input", ""),
        output=output,
    )


def train(
    # Model params
    base_model: str = "",
    # Data params
    data_path: str = "HealthCareMagic-100k.json",
    output_dir: str = "./lora_output",
    # Training params
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    val_set_size: int = 500,
    # LoRA params
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    # Quantization
    use_4bit: bool = True,
    use_8bit: bool = False,
    # Memory optimization
    gradient_checkpointing: bool = True,
    # Training behavior
    train_on_inputs: bool = False,
    group_by_length: bool = False,
    # WandB
    wandb_project: str = "",
    wandb_run_name: str = "",
    # Resume
    resume_from_checkpoint: Optional[str] = None,
    # Config file (overrides other args if provided)
    config_file: Optional[str] = None,
):
    """
    Fine-tune a language model using LoRA/QLoRA.

    Args:
        base_model: Path to base model or HuggingFace model ID.
        data_path: Path to training data (JSON format).
        output_dir: Output directory for checkpoints.
        batch_size: Total batch size (effective = batch_size).
        micro_batch_size: Batch size per device.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate.
        cutoff_len: Maximum sequence length.
        val_set_size: Validation set size.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: LoRA dropout rate.
        lora_target_modules: Modules to apply LoRA to.
        use_4bit: Use 4-bit quantization (QLoRA).
        use_8bit: Use 8-bit quantization.
        gradient_checkpointing: Enable gradient checkpointing.
        train_on_inputs: Include inputs in loss calculation.
        group_by_length: Group similar length sequences.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name.
        resume_from_checkpoint: Path to checkpoint to resume from.
        config_file: Optional config file path (overrides other args).
    """
    # Load config if provided
    if config_file:
        cfg = load_config(config_file)
        base_model = cfg.model.base_model
        data_path = cfg.training.data_path
        output_dir = cfg.training.output_dir
        batch_size = cfg.training.batch_size
        micro_batch_size = cfg.training.micro_batch_size
        num_epochs = cfg.training.num_epochs
        learning_rate = cfg.training.learning_rate
        cutoff_len = cfg.training.cutoff_len
        val_set_size = cfg.training.val_set_size
        lora_r = cfg.training.lora_r
        lora_alpha = cfg.training.lora_alpha
        lora_dropout = cfg.training.lora_dropout
        lora_target_modules = cfg.training.lora_target_modules
        gradient_checkpointing = cfg.training.gradient_checkpointing
        use_4bit = cfg.model.load_in_4bit
        use_8bit = cfg.model.load_in_8bit
        wandb_project = cfg.wandb.project
        wandb_run_name = cfg.wandb.run_name or ""

    # Validation
    if not base_model:
        raise ValueError("Please provide --base_model or set in config file")

    # Set default LoRA target modules for LLaMA
    if lora_target_modules is None:
        lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    # Calculate gradient accumulation
    gradient_accumulation_steps = batch_size // micro_batch_size

    # Handle distributed training
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Device setup
    device_map = "auto"
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("=" * 60)
    logger.info("ChatDoctor LoRA Fine-tuning")
    logger.info("=" * 60)
    logger.info(f"Base model: {base_model}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Batch size: {batch_size} (micro: {micro_batch_size})")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"LoRA r={lora_r}, alpha={lora_alpha}")
    logger.info(f"Quantization: 4-bit={use_4bit}, 8-bit={use_8bit}")
    logger.info("=" * 60)

    # W&B setup
    use_wandb = bool(wandb_project) or "WANDB_PROJECT" in os.environ
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project

    # Quantization config
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

    # Load model
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )

    # Configure LoRA
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # Tokenization function
    def tokenize(prompt: str, add_eos: bool = True) -> dict:
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point: dict) -> dict:
        full_prompt = generate_prompt(data_point, include_output=True)
        tokenized = tokenize(full_prompt)

        if not train_on_inputs:
            user_prompt = generate_prompt(data_point, include_output=False)
            user_tokenized = tokenize(user_prompt, add_eos=False)
            user_len = len(user_tokenized["input_ids"])
            tokenized["labels"] = [-100] * user_len + tokenized["labels"][user_len:]

        return tokenized

    # Load data
    logger.info(f"Loading data from {data_path}...")
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    # Split data
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    logger.info(f"Training samples: {len(train_data)}")
    if val_data:
        logger.info(f"Validation samples: {len(val_data)}")

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        eval_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=200 if val_set_size > 0 else None,
        save_steps=200,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=data_collator,
    )

    # Disable cache for training
    model.config.use_cache = False

    # Compile model if PyTorch 2.0+
    if torch.__version__ >= "2" and sys.platform != "win32":
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    logger.info(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Training complete!")


def main():
    """CLI entry point."""
    fire.Fire(train)


if __name__ == "__main__":
    main()
