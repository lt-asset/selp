from __future__ import annotations

import argparse

from selp.io import iter_jsonl, write_jsonl
from selp.ltl import atomic_props
from selp.plan_evaluation import evaluate_plan_against_formula


def _predicate_set(record: dict, formula: str, field: str | None) -> set[str]:
    if field and field in record:
        return set(record[field])
    if "env_data" in record:
        predicates = set()
        for floor in record["env_data"].values():
            predicates.update(floor.keys())
        return predicates
    return atomic_props(formula)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate newline-separated plans against LTL formulas.")
    parser.add_argument("input", help="Input JSONL")
    parser.add_argument("output", help="Output JSONL")
    parser.add_argument("--plan-field", default="plan")
    parser.add_argument("--formula-field", default="formula")
    parser.add_argument("--predicate-field", default=None)
    args = parser.parse_args()

    results = []
    for record in iter_jsonl(args.input):
        formula = record[args.formula_field]
        plan = record[args.plan_field]
        predicates = _predicate_set(record, formula, args.predicate_field)
        accepted, error = evaluate_plan_against_formula(plan, formula, predicates)
        result = dict(record)
        result["plan_accepted"] = accepted
        result["plan_error"] = error
        results.append(result)

    write_jsonl(results, args.output)


if __name__ == "__main__":
    main()
