"""
ChatDoctor Model - Core model loading and inference functionality.
Supports base models, LoRA adapters, and quantization.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    PeftModel = None

from chatdoctor.core.config import Config, ModelConfig, InferenceConfig

logger = logging.getLogger(__name__)


class ChatDoctorModel:
    """
    ChatDoctor model wrapper for loading and inference.

    Supports:
    - Base LLaMA/Mistral models
    - LoRA adapters via PEFT
    - 4-bit and 8-bit quantization
    - CPU and GPU inference
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
    ):
        """
        Initialize ChatDoctor model.

        Args:
            model_config: Model configuration.
            inference_config: Inference configuration.
        """
        self.model_config = model_config or ModelConfig()
        self.inference_config = inference_config or InferenceConfig()

        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False

    @classmethod
    def from_config(cls, config: Config) -> "ChatDoctorModel":
        """Create model from a Config object."""
        return cls(
            model_config=config.model,
            inference_config=config.inference,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        lora_path: Optional[str] = None,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ) -> "ChatDoctorModel":
        """
        Create and load model from pretrained weights.

        Args:
            model_path: Path to base model.
            lora_path: Optional path to LoRA adapter.
            load_in_4bit: Use 4-bit quantization.
            device_map: Device mapping strategy.

        Returns:
            Loaded ChatDoctorModel instance.
        """
        model_config = ModelConfig(
            base_model=model_path,
            lora_adapter=lora_path,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
        )
        instance = cls(model_config=model_config)
        instance.load()
        return instance

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            logger.warning("Model already loaded, skipping.")
            return

        config = self.model_config
        base_model_path = config.base_model

        logger.info(f"Loading model from: {base_model_path}")

        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)

        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        if not gpu_available:
            logger.warning("CUDA not available, using CPU. This will be slow.")
            torch_dtype = torch.float32

        # Build model kwargs
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": config.device_map if gpu_available else None,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        # Add quantization config
        if gpu_available:
            if config.load_in_4bit:
                logger.info("Using 4-bit quantization (QLoRA compatible)")
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                )
            elif config.load_in_8bit:
                logger.info("Using 8-bit quantization")
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )

        # Load base model
        logger.info("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **model_kwargs,
        )

        # Load LoRA adapter if specified
        if config.lora_adapter and PEFT_AVAILABLE:
            lora_path = Path(config.lora_adapter)
            if lora_path.exists():
                logger.info(f"Loading LoRA adapter from: {lora_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    str(lora_path),
                    torch_dtype=torch_dtype,
                )
            else:
                logger.warning(f"LoRA adapter not found: {lora_path}")
        elif config.lora_adapter and not PEFT_AVAILABLE:
            logger.warning("PEFT not installed, skipping LoRA adapter.")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Determine device
        if hasattr(self.model, "device"):
            self.device = self.model.device
        elif gpu_available:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._loaded = True
        logger.info(f"Model loaded successfully on {self.device}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> str:
        """
        Generate response for a given prompt.

        Args:
            prompt: Input prompt text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling.
            repetition_penalty: Repetition penalty factor.
            do_sample: Whether to use sampling.

        Returns:
            Generated text.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Use provided values or fall back to config
        config = self.inference_config
        max_new_tokens = max_new_tokens or config.max_new_tokens
        temperature = temperature if temperature is not None else config.temperature
        top_p = top_p if top_p is not None else config.top_p
        top_k = top_k if top_k is not None else config.top_k
        repetition_penalty = repetition_penalty or config.repetition_penalty
        do_sample = do_sample if do_sample is not None else config.do_sample

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = outputs[0, input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

    def chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """
        Chat interface with medical assistant persona.

        Args:
            user_message: User's message/question.
            system_prompt: Optional custom system prompt.

        Returns:
            Assistant's response.
        """
        if system_prompt is None:
            system_prompt = (
                "You are ChatDoctor, a helpful and knowledgeable medical AI assistant. "
                "Provide accurate, concise medical information based on the patient's query. "
                "Always recommend consulting a healthcare professional for serious concerns."
            )

        prompt = f"{system_prompt}\n\nPatient: {user_message}\n\nChatDoctor:"

        response = self.generate(prompt)

        # Clean up response
        if "Patient:" in response:
            response = response.split("Patient:")[0].strip()

        return response

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"ChatDoctorModel(base_model='{self.model_config.base_model}', status={status})"
