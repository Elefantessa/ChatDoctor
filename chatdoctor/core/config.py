"""
Configuration management for ChatDoctor.
Handles loading and validation of YAML configuration files.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml


@dataclass
class ModelConfig:
    """Model-related configuration."""
    base_model: str = "./models/llama-base"
    lora_adapter: Optional[str] = None
    torch_dtype: str = "float16"
    device_map: str = "auto"
    load_in_4bit: bool = True
    load_in_8bit: bool = False


@dataclass
class TrainingConfig:
    """Training-related configuration."""
    output_dir: str = "./models/lora_weights"
    data_path: str = "./data/HealthCareMagic-100k.json"
    batch_size: int = 128
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 1
    learning_rate: float = 3e-4
    cutoff_len: int = 512
    val_set_size: int = 500
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True


@dataclass
class InferenceConfig:
    """Inference-related configuration."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class RAGConfig:
    """RAG-related configuration."""
    mode: str = "none"  # none, csv, wiki
    csv_path: str = "./data/healthcare_disease_dataset.csv"
    num_chunks: int = 4
    chunk_size: int = 250


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_file: Optional[str] = None
    rich_console: bool = True


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""
    project: str = "chatdoctor"
    run_name: Optional[str] = None
    watch: bool = False
    log_model: bool = False


@dataclass
class Config:
    """Main configuration class for ChatDoctor."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create configuration from a dictionary."""
        config = cls()

        if "model" in config_dict:
            config.model = ModelConfig(**config_dict["model"])
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        if "inference" in config_dict:
            config.inference = InferenceConfig(**config_dict["inference"])
        if "rag" in config_dict:
            config.rag = RAGConfig(**config_dict["rag"])
        if "logging" in config_dict:
            config.logging = LoggingConfig(**config_dict["logging"])
        if "wandb" in config_dict:
            config.wandb = WandBConfig(**config_dict["wandb"])

        return config

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "inference": self.inference.__dict__,
            "rag": self.rag.__dict__,
            "logging": self.logging.__dict__,
            "wandb": self.wandb.__dict__,
        }

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from file or return default.

    Args:
        config_path: Path to config file. If None, looks for config.yaml in
                     standard locations or returns default config.

    Returns:
        Config object.
    """
    if config_path is not None:
        return Config.from_yaml(config_path)

    # Look for config in standard locations
    search_paths = [
        Path("config.yaml"),
        Path("configs/config.yaml"),
        Path.home() / ".chatdoctor" / "config.yaml",
    ]

    for path in search_paths:
        if path.exists():
            return Config.from_yaml(path)

    # Return default config
    return get_default_config()
