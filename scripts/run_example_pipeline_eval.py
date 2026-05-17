from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from selp.config import DEFAULT_PLAN_PROMPT
from selp.constrained_decoding import constrained_decode_plan_trie
from selp.io import read_jsonl, write_jsonl
from selp.ltl import build_full_traj_states, build_hard_constraint, ltl_to_digraph
from selp.ltl_translation import (
    combine_formulas,
    explain_commands,
    paraphrase_commands,
    read_prompt_template,
    translate_ltl,
    vote_ltl,
)
from selp.plan_evaluation import check_if_plan_can_be_accepted_by_ltl


def load_model(model_path: str, tokenizer_path: str):
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


def first_model_device(model) -> torch.device:
    return next(model.parameters()).device


def generate_plans(
    formula_records: list[dict[str, Any]],
    planner_model_path: str,
    planner_tokenizer_path: str,
    temperature: float,
    max_plan_steps: int,
) -> list[dict[str, Any]]:
    tokenizer, model = load_model(planner_model_path, planner_tokenizer_path)
    plan_records = []
    for record in formula_records:
        result = dict(record)
        if not record["formula_valid"]:
            result.update(
                {
                    "plan_full": "",
                    "plan": "",
                    "plan_error": "combined LTL formula is invalid",
                    "violate_num": None,
                    "tried_num": None,
                }
            )
            plan_records.append(result)
            continue

        predicates = set(record["predicate_set"])
        try:
            dfa, accepting_states, initial_state, is_dba = ltl_to_digraph(
                record["formula"],
                build_full_traj_states(predicates),
                hard_constraint=build_hard_constraint(predicates),
            )
            if dfa is None or accepting_states is None or initial_state is None or is_dba is None:
                raise RuntimeError("empty automaton")
            prompt = DEFAULT_PLAN_PROMPT.format(
                env_data=record["env_data"],
                description=record["description"],
            )
            plan_full, error, violate_num, tried_num = constrained_decode_plan_trie(
                tokenizer=tokenizer,
                model=model,
                predicate_set=predicates,
                dfa=dfa,
                accepting_states=accepting_states,
                initial_state=initial_state,
                is_dba=is_dba,
                user_prompt=prompt,
                temperature=temperature,
                eos_id=tokenizer.eos_token_id,
                pad_id=tokenizer.pad_token_id,
                device=first_model_device(model),
                max_plan_steps=max_plan_steps,
                num_output=1,
            )
            plan_tail = plan_full[len(prompt) :] if plan_full.startswith(prompt) else plan_full
            result.update(
                {
                    "plan_full": plan_full,
                    "plan": plan_tail,
                    "plan_error": error,
                    "violate_num": violate_num,
                    "tried_num": tried_num,
                }
            )
        except Exception as exc:
            result.update(
                {
                    "plan_full": "",
                    "plan": "",
                    "plan_error": f"{type(exc).__name__}: {exc}",
                    "violate_num": None,
                    "tried_num": None,
                }
            )
        plan_records.append(result)
    del model
    torch.cuda.empty_cache()
    return plan_records


def evaluate_plans(plan_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eval_records = []
    for record in plan_records:
        result = dict(record)
        if not record.get("plan"):
            result.update({"plan_accepted": False, "plan_eval_error": "no plan generated"})
            eval_records.append(result)
            continue
        try:
            predicates = set(record["predicate_set"])
            dfa, accepting_states, initial_state, is_dba = ltl_to_digraph(
                record["formula"],
                build_full_traj_states(predicates),
                hard_constraint=build_hard_constraint(predicates),
            )
            plan_for_eval = record["plan"].replace("DONE", "").strip()
            accepted, error = check_if_plan_can_be_accepted_by_ltl(
                plan_for_eval,
                predicates,
                dfa,
                accepting_states,
                initial_state,
                is_dba,
            )
            result.update({"plan_accepted": accepted, "plan_eval_error": error})
        except Exception as exc:
            result.update({"plan_accepted": False, "plan_eval_error": f"{type(exc).__name__}: {exc}"})
        eval_records.append(result)
    return eval_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SELP example pipeline test.")
    parser.add_argument("--input", default="data/example_test_data.jsonl")
    parser.add_argument("--eval-dir", default="eval_log")
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument(
        "--ltl-model-path",
        default="/local2/wu1827_/robot/selp_repo/LLMs/CodeLlama2-ff-ltl_trans_two-checkpoint-600",
    )
    parser.add_argument("--ltl-tokenizer-path", default="codellama/CodeLlama-7b-hf")
    parser.add_argument(
        "--planner-model-path",
        default="/local2/wu1827_/robot/selp_repo/LLMs/llama2-ff-drone-Apr30_four-checkpoint-1539",
    )
    parser.add_argument("--planner-tokenizer-path", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num-ltl-samples", type=int, default=3)
    parser.add_argument("--ltl-temperature", type=float, default=0.6)
    parser.add_argument("--planner-temperature", type=float, default=0.6)
    parser.add_argument("--ltl-max-new-tokens", type=int, default=256)
    parser.add_argument("--max-plan-steps", type=int, default=80)
    parser.add_argument(
        "--explanation-system-prompt-file",
        required=True,
        help="System prompt file for the OpenAI explanation stage.",
    )
    parser.add_argument(
        "--explanation-user-prompt-file",
        required=True,
        help="User prompt template file for the OpenAI explanation stage. Use {payload}.",
    )
    parser.add_argument(
        "--paraphrase-system-prompt-file",
        required=True,
        help="System prompt file for the OpenAI paraphrase stage.",
    )
    parser.add_argument(
        "--paraphrase-user-prompt-file",
        required=True,
        help="User prompt template file for the OpenAI paraphrase stage. Use {explanation_record}.",
    )
    parser.add_argument(
        "--stop-after",
        choices=["openai", "raw_ltl", "voting", "plans", "eval"],
        default="eval",
        help="Stop after a pipeline stage. Useful when ML and Spot run in different environments.",
    )
    args = parser.parse_args()

    start = time.time()
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    data_records = read_jsonl(args.input)

    client = OpenAI()
    explanation_system_prompt = read_prompt_template(args.explanation_system_prompt_file)
    explanation_user_prompt = read_prompt_template(args.explanation_user_prompt_file)
    paraphrase_system_prompt = read_prompt_template(args.paraphrase_system_prompt_file)
    paraphrase_user_prompt = read_prompt_template(args.paraphrase_user_prompt_file)
    explanation_records = [
        explain_commands(
            client,
            args.openai_model,
            record,
            system_prompt=explanation_system_prompt,
            user_prompt_template=explanation_user_prompt,
        )
        for record in data_records
    ]
    write_jsonl(explanation_records, eval_dir / "openai_explanations.jsonl")

    paraphrase_records = [
        paraphrase_commands(
            client,
            args.openai_model,
            explanation_record,
            system_prompt=paraphrase_system_prompt,
            user_prompt_template=paraphrase_user_prompt,
        )
        for explanation_record in explanation_records
    ]
    write_jsonl(paraphrase_records, eval_dir / "openai_paraphrases.jsonl")
    if args.stop_after == "openai":
        print(json.dumps({"stopped_after": "openai", "eval_dir": str(eval_dir)}, indent=2))
        return

    raw_ltl_records = translate_ltl(
        paraphrase_records,
        model_path=args.ltl_model_path,
        tokenizer_path=args.ltl_tokenizer_path,
        num_samples=args.num_ltl_samples,
        temperature=args.ltl_temperature,
        max_new_tokens=args.ltl_max_new_tokens,
    )
    write_jsonl(raw_ltl_records, eval_dir / "ltl_translation_raw.jsonl")
    if args.stop_after == "raw_ltl":
        print(json.dumps({"stopped_after": "raw_ltl", "eval_dir": str(eval_dir)}, indent=2))
        return

    voted_ltl_records = vote_ltl(raw_ltl_records)
    write_jsonl(voted_ltl_records, eval_dir / "ltl_voting.jsonl")

    formula_records = combine_formulas(data_records, voted_ltl_records)
    write_jsonl(formula_records, eval_dir / "combined_formulas.jsonl")
    if args.stop_after == "voting":
        print(json.dumps({"stopped_after": "voting", "eval_dir": str(eval_dir)}, indent=2))
        return

    plan_records = generate_plans(
        formula_records,
        planner_model_path=args.planner_model_path,
        planner_tokenizer_path=args.planner_tokenizer_path,
        temperature=args.planner_temperature,
        max_plan_steps=args.max_plan_steps,
    )
    write_jsonl(plan_records, eval_dir / "generated_plans.jsonl")
    if args.stop_after == "plans":
        print(json.dumps({"stopped_after": "plans", "eval_dir": str(eval_dir)}, indent=2))
        return

    eval_records = evaluate_plans(plan_records)
    write_jsonl(eval_records, eval_dir / "plan_eval.jsonl")

    summary = {
        "input": args.input,
        "eval_dir": str(eval_dir),
        "openai_model": args.openai_model,
        "ltl_model_path": args.ltl_model_path,
        "planner_model_path": args.planner_model_path,
        "num_records": len(data_records),
        "num_ltl_samples": args.num_ltl_samples,
        "elapsed_seconds": round(time.time() - start, 3),
        "num_valid_combined_formulas": sum(1 for record in formula_records if record["formula_valid"]),
        "num_generated_plans": sum(1 for record in plan_records if record.get("plan")),
        "num_accepted_plans": sum(1 for record in eval_records if record.get("plan_accepted")),
    }
    (eval_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
