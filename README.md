# SELP

This repository provides artifacts for the paper ["SELP: Generating Safe and Efficient Task Plans for Robot Agents with
Large Language Models"](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11128420) (ICRA 2025).

## Installation
Clone the repository:
```bash
 git clone https://github.com/lt-asset/selp.git
 ```

Create a fresh conda environment from the repository root:

```bash
conda create -n selp python=3.11 -y
conda activate selp
conda install -c conda-forge spot -y
python -m pip install --upgrade pip
python -m pip install -e ".[ml,dev]"
python -m pip install openai
```

`spot` is required for semantic LTL equivalence and automaton-based plan
checking. Install it from `conda-forge`.

If the default PyTorch wheel does not match your CUDA driver, install the appropriate PyTorch build first, then rerun:

```bash
python -m pip install -e ".[ml,dev]"
```

Check the install:

```bash
python -c "import selp; print('selp import ok')"
python -c "import spot; print('spot import ok')"
```

## Models

Fine-tuned model weights will be released soon.

## Quick Starts

Set your OpenAI key before running the full pipeline:

```bash
export OPENAI_API_KEY=your_api_key
```

The full SELP pipeline runs explanation, paraphrase, local LTL translation,
equivalence voting, constrained plan generation, and final plan evaluation:

```bash
python scripts/run_example_pipeline_eval.py \
  --input data/example_test_data.jsonl \
  --eval-dir eval_log \
  --explanation-system-prompt-file prompts/explanation_system.txt \
  --explanation-user-prompt-file prompts/explanation_user.txt \
  --paraphrase-system-prompt-file prompts/paraphrase_system.txt \
  --paraphrase-user-prompt-file prompts/paraphrase_user.txt \
  --ltl-model-path /path/LLMs/ltl-model \
  --ltl-tokenizer-path codellama/CodeLlama-7b-hf \
  --planner-model-path /path/LLMs/plan-model \
  --planner-tokenizer-path meta-llama/Llama-2-7b-hf \
  --num-ltl-samples 10 \
  --ltl-temperature 0.6 \
  --planner-temperature 0.6 \
  --max-plan-steps 80
```

Pipeline outputs are written under `--eval-dir`:

- `openai_explanations.jsonl`
- `openai_paraphrases.jsonl`
- `ltl_translation_raw.jsonl`
- `ltl_voting.jsonl`
- `combined_formulas.jsonl`
- `generated_plans.jsonl`
- `plan_eval.jsonl`
- `run_summary.json`

To stop after an intermediate stage, add one of these options to the full
pipeline command:

```bash
--stop-after openai
--stop-after raw_ltl
--stop-after voting
--stop-after plans
```

The plan-generation stage follows the online constrained-decoding design: it builds the automaton from the combined LTL formula, then validates each next action during trie-based decoding. 

## Standalone Utilities

Run direct NL-to-LTL inference without the explanation/paraphrase stages:

```bash
python scripts/infer_ltl.py input.jsonl ltl_predictions.jsonl \
  --model-path ./path/LLMs/ltl-model \
  --tokenizer-path codellama/CodeLlama-7b-hf \
  --description-field description \
  --output-field model_output \
  --num-return-sequences 10
```

Run semantic equivalence voting over generated LTL candidates:

```bash
python scripts/equivalence_vote.py predictions.jsonl voted.jsonl --formula-field formula --candidates-field model_output
```

Vote over raw full-pipeline LTL samples and combine component formulas:

```bash
python scripts/vote_and_combine_ltl.py --input data/example_test_data.jsonl --eval-dir eval_log
```

Generate plans with constrained decoding:

```bash
python scripts/constrained_plan_generate.py combined_formulas.jsonl plans.jsonl \
  --model-path /path/LLMs/plan-model \
  --tokenizer-path meta-llama/Llama-2-7b-hf \
  --formula-field formula \
  --description-field description \
  --env-field env_data
```

## Citation

```
@inproceedings{wu2025selp,
  title={SELP: Generating safe and efficient task plans for robot agents with large language models},
  author={Wu, Yi and Xiong, Zikang and Hu, Yiran and Iyengar, Shreyash S and Jiang, Nan and Bera, Aniket and Tan, Lin and Jagannathan, Suresh},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={2599--2605},
  year={2025},
  organization={IEEE}
}
```
