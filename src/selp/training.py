"""Training and dataset preparation helpers for causal LM fine-tuning."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def configure_tokenizer(tokenizer):
    """Apply the padding conventions used by the development training scripts."""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_instruction_rows(
    rows: Iterable[dict[str, str]],
    tokenizer,
    max_length: int = 1024,
) -> dict[str, list[list[int]]]:
    """Tokenize prompt/completion rows into causal-LM input and label arrays."""

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    output = {"input_ids": [], "labels": []}

    for row in rows:
        prompt_ids = tokenizer.encode(row["prompt"])
        completion_ids = tokenizer.encode(row["completion"])
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids

        if len(input_ids) > max_length:
            continue

        input_ids += [pad_id] * (max_length - len(input_ids))
        labels += [-100] * (max_length - len(labels))
        output["input_ids"].append(input_ids)
        output["labels"].append(labels)

    return output


class CausalLMDataset:
    """Thin torch Dataset wrapper for tokenized Hugging Face datasets."""

    def __init__(self, data, pad_token_id: int):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for training") from exc

        class _Dataset(torch.utils.data.Dataset):
            def __init__(self, data, pad_token_id: int):
                self.data = data
                self.pad_token_id = pad_token_id

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                item = self.data[index]
                input_ids = item["input_ids"]
                labels = item["labels"]
                attention_mask = input_ids.ne(self.pad_token_id).to(dtype=torch.bool)
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

        self.dataset = _Dataset(data, pad_token_id)


def train_causal_lm(
    model_name_or_path: str,
    train_data_path: str,
    output_dir: str,
    validation_data_path: str | None = None,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-5,
    num_train_epochs: float = 2.0,
    warmup_steps: int = 200,
    logging_steps: int = 200,
    save_strategy: str = "epoch",
    bf16: bool = True,
    deepspeed: str | None = None,
) -> None:
    """Train a causal language model from pre-tokenized datasets."""

    try:
        import torch
        from datasets import load_from_disk
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install the `ml` extra to train models: pip install -e '.[ml]'") from exc

    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(model_name_or_path))
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()

    train_data = load_from_disk(train_data_path)
    train_data.set_format(type="torch", columns=["input_ids", "labels"])
    train_dataset = CausalLMDataset(train_data, tokenizer.pad_token_id).dataset

    eval_dataset = None
    if validation_data_path:
        validation_data = load_from_disk(validation_data_path)
        validation_data.set_format(type="torch", columns=["input_ids", "labels"])
        eval_dataset = CausalLMDataset(validation_data, tokenizer.pad_token_id).dataset

    trainer_args: dict[str, Any] = {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_steps": warmup_steps,
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "num_train_epochs": num_train_epochs,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "save_strategy": save_strategy,
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "prediction_loss_only": True,
        "bf16": bf16,
        "ddp_find_unused_parameters": False,
    }
    if eval_dataset is not None:
        trainer_args.update({"evaluation_strategy": "steps", "eval_steps": logging_steps})
    if deepspeed:
        trainer_args["deepspeed"] = deepspeed

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**trainer_args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
