from __future__ import annotations

import argparse

from selp.config import DEFAULT_PLAN_BASE_MODEL, DEFAULT_PLAN_MODEL_PATH, DEFAULT_PLAN_PROMPT
from selp.constrained_decoding import constrained_decode_plan_trie
from selp.io import iter_jsonl, write_jsonl
from selp.ltl import atomic_props, build_full_traj_states, build_hard_constraint, ltl_to_digraph
from selp.modeling import load_causal_lm_and_tokenizer


def _predicate_set(record: dict, formula: str) -> set[str]:
    if "env_data" in record:
        predicates = set()
        for rooms in record["env_data"].values():
            predicates.update(rooms.keys())
        return predicates
    return atomic_props(formula)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plans with trie-based constrained decoding.")
    parser.add_argument("input", help="Input JSONL")
    parser.add_argument("output", help="Output JSONL")
    parser.add_argument("--model-path", default=DEFAULT_PLAN_MODEL_PATH)
    parser.add_argument("--tokenizer-path", default=DEFAULT_PLAN_BASE_MODEL)
    parser.add_argument("--formula-field", default="formula")
    parser.add_argument("--description-field", default="description")
    parser.add_argument("--env-field", default="env_data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-plan-steps", type=int, default=500)
    args = parser.parse_args()

    model, tokenizer = load_causal_lm_and_tokenizer(args.model_path, args.tokenizer_path, args.device)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    results = []
    for record in iter_jsonl(args.input):
        formula = record[args.formula_field]
        predicates = _predicate_set(record, formula)
        dfa, accepting_states, initial_state, is_dba = ltl_to_digraph(
            formula,
            build_full_traj_states(predicates),
            hard_constraint=build_hard_constraint(predicates),
        )

        result = dict(record)
        if dfa is None or accepting_states is None or initial_state is None or is_dba is None:
            result["constraint_decoding"] = [["", "formula automaton is empty", 0, 0]]
            results.append(result)
            continue

        prompt = DEFAULT_PLAN_PROMPT.format(
            env_data=record.get(args.env_field, {}),
            description=record[args.description_field],
        )
        plan, error, violate_num, tried_num = constrained_decode_plan_trie(
            tokenizer=tokenizer,
            model=model,
            predicate_set=predicates,
            dfa=dfa,
            accepting_states=accepting_states,
            initial_state=initial_state,
            is_dba=is_dba,
            user_prompt=prompt,
            temperature=args.temperature,
            eos_id=eos_id,
            pad_id=pad_id,
            device=args.device,
            max_plan_steps=args.max_plan_steps,
        )
        result["constraint_decoding"] = [[plan, error, violate_num, tried_num]]
        results.append(result)

    write_jsonl(results, args.output)


if __name__ == "__main__":
    main()
