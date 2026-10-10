# Secure Code Generation with Reasoning Models

This repository contains the code for the paper "SecPI: Secure Code Generation with Reasoning Models via Security Reasoning Internalization". It supports dataset construction, model fine-tuning, and evaluation across three security-aware code generation benchmarks.

## Setup

```bash
pip install -r requirements.txt
```

Set API credentials as environment variables:
```bash
export OPENAI_API_KEY="<your-key>"
export OPENAI_BASE_URL="<your-base-url>"   # optional, for litellm proxy or custom endpoints
```

## Pipeline Overview

The full pipeline has three stages:

1. **Dataset Generation** — filter KodCode-V1 for security-relevant samples, create subsets, and generate reasoning traces with an LLM
2. **Training** — fine-tune a model on the generated dataset using LLaMA-Factory
3. **Evaluation** — evaluate the fine-tuned model on CWEval, LiveCodeBench, and BaxBench

## Stage 1: Dataset Generation

The script `scripts/run_generate_dataset.sh` runs the full dataset generation pipeline:

```bash
bash scripts/run_generate_dataset.sh
```

This script performs the following steps:

1. **Filter** 10% of KodCode-V1 using GPT-4o to identify security-relevant samples. Produces two splits:
   - `data/dataset_security/KodCode-KodCode-V1_security_relevant.json`
   - `data/dataset_security/KodCode-KodCode-V1_security_irrelevant.json`

2. **Create subsets** of the security-relevant dataset:
   - Fraction-based: 11%, 22%, 33%, 67% random samples
   - Category-based: injection-only (no memory CWEs) and memory-only (no injection CWEs)

3. **Generate reasoning traces** on the relevant dataset using DeepSeek-R1 with three prompt types:
   - `secure` — generic security reminder
   - `secure+` — CWE identification and mitigation reasoning
   - `cwe` — per-sample CWE-specific hints

4. **Generate reasoning traces** on the irrelevant dataset using `secure+` prompt (1324 random samples)

Generated datasets are saved under `data/dataset_generated/`.

### Individual Commands

You can also run each step individually:

```bash
# Filter dataset
python -m src --mode filter-dataset \
    --checker_model openai/gpt-4o --checker_provider openai \
    --sample_fraction 0.1

# Create subsets
python scripts/subset_dataset.py fraction <input.json> <output.json> --frac 0.33
python scripts/subset_dataset.py category <input.json> <output.json> --include injection --exclude memory

# Generate traces
python -m src --mode generate-dataset \
    --model <model> --provider <provider> \
    --gen_input <input.json> --gen_prompt secure+
```

Available `--gen_prompt` options: `none`, `direct`, `secure`, `secure+`, `cwe`.

## Stage 2: Training

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for fine-tuning. Example training configs are in `config/`.

### Setup

```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### Register the dataset

Add your generated dataset to `LLaMA-Factory/data/dataset_info.json`:
```json
{
  "kodcode-v1": {
    "file_name": "<path-to-generated-dataset>/generated.json",
    "columns": {
      "prompt": "question",
      "response": "generated_solution"
    }
  }
}
```

### Run training

```bash
llamafactory-cli train ../config/kodcode-v1-qwq.yaml
```

The example config `config/kodcode-v1-qwq.yaml` trains QwQ-32B with full SFT using DeepSpeed ZeRO-3. Key parameters:
- Context length: 16,384 tokens
- Learning rate: 1e-5 with cosine scheduler
- 3 epochs, batch size 48
- BF16 precision

Modify `model_name_or_path`, `dataset`, `template`, and `output_dir` in the config to match the setup in the paper.

## Stage 3: Evaluation

### Start a vLLM server (for local models)

```bash
vllm serve <model-path> --port 8000
```

### Run evaluation

The script `run_eval.sh` runs the full evaluation pipeline (generate, test, summarize) across benchmarks:

```bash
# Evaluate on all benchmarks with vLLM
./run_eval.sh --model <model-name> --provider vllm

# Evaluate on CWEval only, with secure+ prompt
./run_eval.sh --model <model-name> --provider vllm \
    --benchmarks cweval --prompts "direct secure+"

# Skip generation, only run tests and summary
./run_eval.sh --model <model-name> --provider vllm --skip_generate
```

Supported benchmarks: `cweval`, `livecodebench`, `baxbench`.
Supported providers: `vllm`, `openai`, `together`, `openrouter`, `anthropic`.
Supported prompt types: `direct`, `secure`, `secure+` (our prompt in the paper).

Results are saved to `eval-traces/` by default (configurable with `--results_dir`).

### Trace Analysis

Analyze security reasoning in model traces:

```bash
python -m src --mode analyze-traces --benchmark cweval \
    --results_dir eval-traces/<model>@<prompt>@<n>@<temp>
```

## Project Structure

```
config/                  Training configs for LLaMA-Factory
scripts/
  run_generate_dataset.sh   Dataset generation pipeline
  subset_dataset.py         Subset creation utility
src/
  main.py                   Unified CLI entry point
  filter_dataset.py         KodCode-V1 security filtering
  generate_dataset.py       Reasoning trace generation
  analyze_traces.py         Security trace analysis
  benchmarks/               CWEval, LiveCodeBench, BaxBench tasks and prompts
  inference_models/         LLM provider abstractions (OpenAI, Anthropic, vLLM, etc.)
run_eval.sh                 Evaluation pipeline script
sec_keywords.json           General security keywords
cwe_keywords.json           Per-CWE keyword mappings
```

## Keywords and Prompts

- `sec_keywords.json` — keywords used to detect security reasoning in traces
- `cwe_keywords.json` — per-CWE keyword mappings for keyword coverage metrics
- `src/benchmarks/prompts.py` — prompt templates for all benchmarks and prompt types
