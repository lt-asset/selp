from __future__ import annotations

import argparse

from selp.io import iter_jsonl, write_jsonl
from selp.ltl import are_equivalent, is_valid_formula


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted LTL formulas with Spot equivalence.")
    parser.add_argument("input", help="Input JSONL")
    parser.add_argument("output", help="Output JSONL")
    parser.add_argument("--prediction-field", default="majority_formula")
    parser.add_argument("--formula-field", default="formula")
    args = parser.parse_args()

    results = []
    for index, record in enumerate(iter_jsonl(args.input)):
        prediction = record.get(args.prediction_field)
        ground_truth = record.get(args.formula_field) or record.get("ground_truth_formula")
        result = dict(record)
        result["eval_index"] = index

        if not prediction or not ground_truth:
            result["ltl_equal"] = False
            result["ltl_error"] = "missing prediction or ground truth"
        elif not is_valid_formula(prediction):
            result["ltl_equal"] = False
            result["ltl_error"] = "prediction syntax error"
        else:
            result["ltl_equal"] = are_equivalent(prediction, ground_truth)
            result["ltl_error"] = ""
        results.append(result)

    write_jsonl(results, args.output)


if __name__ == "__main__":
    main()
