from __future__ import annotations

import argparse

from selp.equivalence_voting import run_equivalence_voting


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic equivalence voting over LTL predictions.")
    parser.add_argument("input", help="Input JSONL with generated formulas")
    parser.add_argument("output", help="Output JSONL with voting fields")
    parser.add_argument("--ground-truth", default=None, help="Optional JSON/JSONL ground-truth file")
    parser.add_argument("--id-field", default="task_ID")
    parser.add_argument("--formula-field", default="formula")
    parser.add_argument("--candidates-field", default=None)
    args = parser.parse_args()

    run_equivalence_voting(
        args.input,
        args.output,
        ground_truth_path=args.ground_truth,
        id_field=args.id_field,
        formula_field=args.formula_field,
        candidates_field=args.candidates_field,
    )


if __name__ == "__main__":
    main()
