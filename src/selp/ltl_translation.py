"""NL-to-LTL translation helpers.

This module contains the reusable pieces of the LTL translation pipeline:
OpenAI-based explanation/paraphrase, local LTL-model sampling, and formula
voting/combination.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .equivalence_voting import majority_vote
from .ltl import is_valid_formula


DEFAULT_EXPLANATION_SYSTEM_PROMPT = (
    "You explain temporal-logic navigation commands for a robot. "
    "Preserve all symbolic location names exactly, including underscores. "
    "Return JSON only."
)

DEFAULT_EXPLANATION_USER_PROMPT = (
    "For each command, write one concise explanation of its temporal "
    "meaning. Return {\"items\": [{\"id\": ..., \"explanation\": ...}]} "
    "with the same item ids.\n\n"
    "{payload}"
)

DEFAULT_PARAPHRASE_SYSTEM_PROMPT = (
    "You paraphrase robot navigation commands for LTL translation. "
    "Keep the meaning unchanged. Preserve all symbolic location names "
    "exactly, including underscores. Return JSON only."
)

DEFAULT_PARAPHRASE_USER_PROMPT = (
    "For each item, produce one clearer paraphrase that keeps the same "
    "temporal meaning. Return {\"items\": [{\"id\": ..., "
    "\"paraphrase\": ...}]} with the same item ids.\n\n"
    "{explanation_record}"
)


def read_prompt_template(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8")


def render_prompt_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def commands_for_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    commands = [{"id": "main_task", "kind": "main_task", "text": record["main_task_dsp"]}]
    for index, constraint in enumerate(record["constraint_dsp"], start=1):
        commands.append(
            {
                "id": f"constraint_{index}",
                "kind": "constraint",
                "constraint_index": index,
                "text": constraint,
            }
        )
    return commands


def openai_json(client: Any, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("OpenAI returned an empty response")
    return json.loads(content)


def explain_commands(
    client: Any,
    model: str,
    record: dict[str, Any],
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
) -> dict[str, Any]:
    commands = commands_for_record(record)
    payload = {"test_idx": record["test_idx"], "commands": commands}
    payload_json = json.dumps(payload, ensure_ascii=False)
    result = openai_json(
        client,
        model,
        [
            {
                "role": "system",
                "content": system_prompt or DEFAULT_EXPLANATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": render_prompt_template(
                    user_prompt_template or DEFAULT_EXPLANATION_USER_PROMPT,
                    {"payload": payload_json},
                ),
            },
        ],
    )
    explanation_by_id = {item["id"]: item["explanation"] for item in result["items"]}
    return {
        "test_idx": record["test_idx"],
        "items": [
            {
                **command,
                "explanation": explanation_by_id[command["id"]],
            }
            for command in commands
        ],
    }


def paraphrase_commands(
    client: Any,
    model: str,
    explanation_record: dict[str, Any],
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
) -> dict[str, Any]:
    explanation_record_json = json.dumps(explanation_record, ensure_ascii=False)
    result = openai_json(
        client,
        model,
        [
            {
                "role": "system",
                "content": system_prompt or DEFAULT_PARAPHRASE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": render_prompt_template(
                    user_prompt_template or DEFAULT_PARAPHRASE_USER_PROMPT,
                    {"explanation_record": explanation_record_json},
                ),
            },
        ],
    )
    paraphrase_by_id = {item["id"]: item["paraphrase"] for item in result["items"]}
    return {
        "test_idx": explanation_record["test_idx"],
        "items": [
            {
                **item,
                "paraphrase": paraphrase_by_id[item["id"]],
            }
            for item in explanation_record["items"]
        ],
    }


def load_ltl_model(model_path: str, tokenizer_path: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the `ml` extra for model inference: pip install -e '.[ml]'") from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    return tokenizer, model


def first_model_device(model) -> Any:
    return next(model.parameters()).device


def generate_ltl_text(
    tokenizer,
    model,
    prompt: str,
    num_return_sequences: int,
    temperature: float,
    max_new_tokens: int,
) -> list[str]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for generation") from exc

    device = first_model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if temperature <= 0:
        generation_kwargs = {
            "do_sample": False,
            "num_beams": max(1, num_return_sequences),
            "num_return_sequences": num_return_sequences,
        }
    else:
        generation_kwargs = {
            "do_sample": True,
            "temperature": temperature,
            "top_k": 0,
            "top_p": 1.0,
            "num_return_sequences": num_return_sequences,
        }
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
            max_new_tokens=max_new_tokens,
            remove_invalid_values=True,
            renormalize_logits=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    outputs = []
    for generated in output_ids:
        text = tokenizer.decode(
            generated[inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        outputs.append(clean_formula_or_plan(text))
    return outputs


def clean_formula_or_plan(text: str) -> str:
    text = text.strip()
    text = text.replace("```ltl", "").replace("```", "").strip()
    if "\n" in text:
        first_nonempty = next((line.strip() for line in text.splitlines() if line.strip()), "")
        return first_nonempty
    return text.strip().strip('"')


def translate_ltl(
    paraphrase_records: list[dict[str, Any]],
    model_path: str,
    tokenizer_path: str,
    num_samples: int,
    temperature: float,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for LTL translation") from exc

    tokenizer, model = load_ltl_model(model_path, tokenizer_path)
    output_records = []
    for record in paraphrase_records:
        translated_items = []
        for item in record["items"]:
            prompt = f"Task Description:\n{item['paraphrase']}\nPlease write the LTL formula:\n"
            samples = generate_ltl_text(
                tokenizer,
                model,
                prompt,
                num_return_sequences=num_samples,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            translated_items.append({**item, "prompt": prompt, "model_output": samples})
        output_records.append({"test_idx": record["test_idx"], "items": translated_items})
    del model
    torch.cuda.empty_cache()
    return output_records


def vote_ltl(raw_ltl_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    voted_records = []
    for record in raw_ltl_records:
        items = []
        for item in record["items"]:
            vote = majority_vote(item["model_output"])
            items.append({**item, **vote})
        voted_records.append({"test_idx": record["test_idx"], "items": items})
    return voted_records


def room_predicates(record: dict[str, Any]) -> set[str]:
    predicates: set[str] = set()
    for rooms in record["env_data"].values():
        predicates.update(rooms.keys())
    return predicates


def combine_formulas(data_records: list[dict[str, Any]], voted_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    data_by_idx = {record["test_idx"]: record for record in data_records}
    combined = []
    for vote_record in voted_records:
        formulas = []
        invalid_items = []
        for item in vote_record["items"]:
            formula = item["majority_formula"]
            if formula == "SyntaxError" or not is_valid_formula(formula):
                invalid_items.append({"id": item["id"], "formula": formula})
            else:
                formulas.append(f"({formula})")
        full_formula = " & ".join(formulas) if formulas else ""
        combined.append(
            {
                "test_idx": vote_record["test_idx"],
                "description": data_by_idx[vote_record["test_idx"]]["description"],
                "env_data": data_by_idx[vote_record["test_idx"]]["env_data"],
                "predicate_set": sorted(room_predicates(data_by_idx[vote_record["test_idx"]])),
                "component_formulas": [
                    {"id": item["id"], "formula": item["majority_formula"]}
                    for item in vote_record["items"]
                ],
                "invalid_items": invalid_items,
                "formula": full_formula,
                "formula_valid": bool(full_formula) and is_valid_formula(full_formula),
            }
        )
    return combined
