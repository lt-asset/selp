"""Hugging Face model loading and generation helpers."""

from __future__ import annotations

from .training import configure_tokenizer


def load_causal_lm_and_tokenizer(model_path: str, tokenizer_path: str | None = None, device: str = "cuda:0"):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the `ml` extra for model inference: pip install -e '.[ml]'") from exc

    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(tokenizer_path or model_path))
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model = model.to(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    model.eval()
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda:0",
    temperature: float = 0.6,
    max_new_tokens: int = 512,
    num_return_sequences: int = 1,
) -> list[str]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for generation") from exc

    actual_device = device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(actual_device)
    if temperature <= 0:
        generation_kwargs = {
            "do_sample": False,
            "num_beams": max(1, num_return_sequences),
            "num_return_sequences": num_return_sequences,
        }
    else:
        generation_kwargs = {
            "do_sample": True,
            "top_k": 0,
            "top_p": 1.0,
            "temperature": temperature,
            "num_return_sequences": num_return_sequences,
        }
    output_ids = model.generate(
        inputs=input_ids,
        max_new_tokens=max_new_tokens,
        **generation_kwargs,
        remove_invalid_values=True,
        renormalize_logits=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    outputs = []
    for generated_id in output_ids:
        outputs.append(
            tokenizer.decode(
                generated_id[input_ids.size(1):],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
        )
    return outputs
