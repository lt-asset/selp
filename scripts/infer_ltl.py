from __future__ import annotations

import argparse

from selp.config import DEFAULT_LTL_BASE_MODEL, DEFAULT_LTL_MODEL_PATH, DEFAULT_LTL_PROMPT
from selp.io import iter_jsonl, write_jsonl
from selp.modeling import generate_text, load_causal_lm_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LTL formulas from natural-language descriptions.")
    parser.add_argument("input", help="Input JSONL")
    parser.add_argument("output", help="Output JSONL")
    parser.add_argument("--model-path", default=DEFAULT_LTL_MODEL_PATH)
    parser.add_argument("--tokenizer-path", default=DEFAULT_LTL_BASE_MODEL)
    parser.add_argument("--description-field", default="description")
    parser.add_argument("--output-field", default="model_output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num-return-sequences", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    model, tokenizer = load_causal_lm_and_tokenizer(args.model_path, args.tokenizer_path, args.device)
    results = []
    for record in iter_jsonl(args.input):
        description = record[args.description_field]
        prompt = DEFAULT_LTL_PROMPT.format(description=description)
        outputs = generate_text(
            model,
            tokenizer,
            prompt,
            device=args.device,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_return_sequences,
        )
        result = dict(record)
        result[args.output_field] = outputs
        results.append(result)

    write_jsonl(results, args.output)


if __name__ == "__main__":
    main()
