from __future__ import annotations

import argparse
from pathlib import Path

from selp.io import read_jsonl, write_jsonl
from selp.ltl_translation import combine_formulas, vote_ltl


def main() -> None:
    parser = argparse.ArgumentParser(description="Vote over raw LTL samples and combine component formulas.")
    parser.add_argument("--input", default="data/example_test_data.jsonl")
    parser.add_argument("--eval-dir", default="eval_log")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    data_records = read_jsonl(args.input)
    raw_ltl_records = read_jsonl(eval_dir / "ltl_translation_raw.jsonl")
    voted_ltl_records = vote_ltl(raw_ltl_records)
    write_jsonl(voted_ltl_records, eval_dir / "ltl_voting.jsonl")
    formula_records = combine_formulas(data_records, voted_ltl_records)
    write_jsonl(formula_records, eval_dir / "combined_formulas.jsonl")
    print(f"wrote {eval_dir / 'ltl_voting.jsonl'}")
    print(f"wrote {eval_dir / 'combined_formulas.jsonl'}")


if __name__ == "__main__":
    main()
