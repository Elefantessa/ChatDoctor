"""Tests for ChatDoctor configuration module."""

import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatdoctor.core.config import (
    Config,
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    load_config,
    get_default_config,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_defaults(self):
        config = ModelConfig()
        assert config.base_model == "./models/llama-base"
        assert config.load_in_4bit is True
        assert config.device_map == "auto"

    def test_custom_values(self):
        config = ModelConfig(
            base_model="meta-llama/Llama-2-7b-hf",
            load_in_4bit=False,
        )
        assert config.base_model == "meta-llama/Llama-2-7b-hf"
        assert config.load_in_4bit is False


class TestConfig:
    """Tests for main Config class."""

    def test_defaults(self):
        config = Config()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.inference, InferenceConfig)

    def test_from_dict(self):
        config_dict = {
            "model": {
                "base_model": "test-model",
                "load_in_4bit": False,
            },
            "training": {
                "num_epochs": 3,
                "learning_rate": 1e-4,
            },
        }

        config = Config.from_dict(config_dict)
        assert config.model.base_model == "test-model"
        assert config.model.load_in_4bit is False
        assert config.training.num_epochs == 3
        assert config.training.learning_rate == 1e-4

    def test_to_dict(self):
        config = Config()
        config_dict = config.to_dict()

        assert "model" in config_dict
        assert "training" in config_dict
        assert "inference" in config_dict
        assert config_dict["model"]["base_model"] == "./models/llama-base"

    def test_yaml_roundtrip(self):
        original = Config()
        original.model.base_model = "test-model-path"
        original.training.num_epochs = 5

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            original.save_yaml(yaml_path)

            loaded = Config.from_yaml(yaml_path)

            assert loaded.model.base_model == "test-model-path"
            assert loaded.training.num_epochs == 5


class TestLoadConfig:
    """Tests for load_config function."""

    def test_default_config(self):
        config = load_config()
        assert isinstance(config, Config)

    def test_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            default_config = get_default_config()
            default_config.model.base_model = "custom-path"
            default_config.save_yaml(config_path)

            loaded = load_config(config_path)
            assert loaded.model.base_model == "custom-path"
