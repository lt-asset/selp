"""Shared configuration defaults for the clean SELP release."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_LTL_MODEL_PATH = "checkpoints/nl_to_ltl_model"
DEFAULT_PLAN_MODEL_PATH = "checkpoints/plan_generation_model"

DEFAULT_LTL_BASE_MODEL = "codellama/CodeLlama-7b-hf"
DEFAULT_PLAN_BASE_MODEL = "meta-llama/Meta-Llama-3-8B"

DEFAULT_LTL_PROMPT = "Task Description:\n{description}\nPlease write the LTL formula:\n"
DEFAULT_PLAN_PROMPT = (
    "Environment Info: {env_data}\n"
    "{description}\n"
    "Please generate a safe and efficient plan:\n"
)


@dataclass(frozen=True)
class GenerationConfig:
    """Model generation settings used by inference scripts."""

    model_path: str
    base_model: str
    device: str = "cuda:0"
    temperature: float = 0.6
    max_new_tokens: int = 512
    num_return_sequences: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    """Common training hyperparameters."""

    model_name_or_path: str
    train_data: str
    output_dir: str
    validation_data: str | None = None
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    num_train_epochs: float = 2.0
    warmup_steps: int = 200
    logging_steps: int = 200
    save_strategy: str = "epoch"
    bf16: bool = True
    deepspeed: str | None = None
