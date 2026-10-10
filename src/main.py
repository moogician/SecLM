"""
Unified generation and evaluation entry point for CWEval, LiveCodeBench, and BaxBench.

Supports four prompt types:
  - direct/none:   Baseline prompt with no security guidance
  - secure:        Adds generic security reminder
  - secure+:       Adds CWE identification + mitigation reasoning
  - cwe/oracle:    CWE-specific / oracle prompting (per-problem hints where available)

Usage examples:

  # CWEval generation with secure+ prompt
  python -m src.main --mode generate --benchmark cweval \\
      --model gpt-4o --prompt secure+ --provider openai --n_samples 5

  # LiveCodeBench generation with direct prompt via vLLM
  python -m src.main --mode generate --benchmark livecodebench \\
      --model deepseek-ai/DeepSeek-R1 --prompt direct --provider vllm

  # BaxBench generation with oracle/specific prompt
  python -m src.main --mode generate --benchmark baxbench \\
      --model gpt-4o --prompt oracle --provider openai

  # Run Docker-based tests on CWEval generated code
  python -m src.main --mode test --benchmark cweval

  # Run Docker-based tests on BaxBench generated code
  python -m src.main --mode test --benchmark baxbench \\
      --model gpt-4o --timeout 60

  # Summarize LiveCodeBench results
  python -m src.main --mode summary --benchmark livecodebench

  # Filter KodCode-V1 dataset for security-relevant samples
  python -m src.main --mode filter-dataset \
      --checker_model gpt-4o --checker_provider openai

  # Analyze trace security metrics
  python -m src.main --mode analyze-traces --benchmark cweval \
      --results_dir eval-traces-final/CWEval/model@prompt@n@temp

  # Generate reasoning+solution dataset from filtered samples
  python -m src.main --mode generate-dataset \
      --model deepseek-ai/DeepSeek-R1 --provider vllm

  # With security prompt appended
  python -m src.main --mode generate-dataset \
      --model gpt-4o --provider openai --gen_prompt secure+

  # Custom input/output paths
  python -m src.main --mode generate-dataset \
      --model gpt-4o --provider openai \
      --gen_input data/my_filtered.json --gen_output data/my_output/
"""

import argparse
import inspect
import logging
from pathlib import Path

from .inference_models import from_cfg
from .benchmarks import from_cfg as benchmark_from_cfg
from .filter_dataset import filter_dataset
from .analyze_traces import analyze_traces
from .generate_dataset import generate_dataset

logging.getLogger("httpx").setLevel(logging.WARNING)


class EvalEntryPoint:
    def __init__(self, cfg):
        self.cfg = cfg
        self.task = benchmark_from_cfg(cfg)
        print(f"Benchmark: {cfg.benchmark}")

    def generate(self):
        self.model = from_cfg(self.cfg)
        print(f"Generating on {self.task} using {self.model}")
        self.task.generate([self.model])

    def test(self):
        print(f"Running Docker-based tests for {self.task}")
        self.task.test([])

    def summary(self):
        benchmark = self.task
        print(f"Summarizing {benchmark} with {benchmark.metric()} metric")
        benchmark.summary()


valid_phases = [
    name
    for name, _ in inspect.getmembers(EvalEntryPoint, predicate=inspect.isfunction)
    if not name.startswith("_")
]

# Standalone modes that don't need EvalEntryPoint
STANDALONE_MODES = {
    "filter-dataset": filter_dataset,
    "analyze-traces": analyze_traces,
    "generate-dataset": generate_dataset,
}


def main(cfg):
    all_modes = valid_phases + list(STANDALONE_MODES.keys())
    assert cfg.mode in all_modes, (
        f"Valid modes: {', '.join(all_modes)}, not {cfg.mode}"
    )
    print(f"Mode: {cfg.mode}")

    if cfg.mode in STANDALONE_MODES:
        STANDALONE_MODES[cfg.mode](cfg)
    else:
        entrypoint = EvalEntryPoint(cfg)
        getattr(entrypoint, cfg.mode)()


def _main():
    parser = argparse.ArgumentParser(
        description="Unified generation and evaluation for CWEval, LiveCodeBench, and BaxBench"
    )
    parser.add_argument(
        "--model", type=str, help="Model name (e.g. gpt-4o, deepseek-ai/DeepSeek-R1)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate", "test", "summary", "filter-dataset", "analyze-traces", "generate-dataset"],
        required=True,
        help="Mode: generate, test, summary, filter-dataset, analyze-traces, or generate-dataset",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        choices=["cweval", "livecodebench", "baxbench"],
        help="Benchmark to run (required for generate/test/summary/analyze-traces)",
    )

    # --- Prompt type ---
    parser.add_argument(
        "--prompt",
        default="direct",
        type=str,
        help="Prompt type: direct/none (baseline), secure, secure+ (CWE reasoning), cwe/oracle (per-problem CWE hints)",
    )

    # --- Generation parameters ---
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for sampling (default: 0.6)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=24000,
        help="Max completion tokens (default: 24000)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of samples per problem (default: 5)",
    )
    parser.add_argument(
        "--max_concurrent_runs",
        type=int,
        default=32,
        help="Maximum number of concurrent runs (default: 32)",
    )

    # --- Provider / inference ---
    parser.add_argument(
        "--provider",
        type=str,
        default="vllm",
        help="Model provider: openai, together, openrouter, anthropic, vllm (default: vllm)",
    )
    parser.add_argument(
        "--vllm_port",
        type=int,
        default=8000,
        help="Port for vLLM server (default: 8000)",
    )

    # --- Evaluation parameters ---
    parser.add_argument(
        "--version", type=str, default=None, help="Dataset version tag"
    )
    parser.add_argument(
        "--eval_subset", default="valid", type=str, help="Evaluation subset"
    )
    parser.add_argument(
        "--eval_setting", default="all", type=str, help="Evaluation setting filter"
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("eval-traces"),
        help="Directory to save results (default: eval-traces)",
    )

    # --- BaxBench-specific ---
    parser.add_argument(
        "--spec_type",
        choices=["openapi", "text"],
        default="openapi",
        help="BaxBench specification type (default: openapi)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="BaxBench: specific scenarios to run",
    )
    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=None,
        help="BaxBench: specific environments to run",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning effort for o1/o3 models (default: high)",
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help="Route requests through OpenRouter (BaxBench)",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use vLLM for generation (BaxBench)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix for result directory (BaxBench)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if results exist",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for test runs in seconds (BaxBench, default: 300)",
    )

    # --- filter-dataset parameters ---
    parser.add_argument(
        "--checker_model",
        type=str,
        default=None,
        help="Model name for security checker (filter-dataset mode)",
    )
    parser.add_argument(
        "--checker_provider",
        type=str,
        default=None,
        help="Provider for security checker (filter-dataset mode)",
    )
    parser.add_argument(
        "--checker_reasoning",
        action="store_true",
        help="Whether the checker model uses reasoning (filter-dataset mode)",
    )
    parser.add_argument(
        "--checker_reasoning_effort",
        type=str,
        default=None,
        help="Reasoning effort for the checker model (filter-dataset mode)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        nargs="+",
        default=None,
        help="KodCode-V1 subsets to filter (e.g. Filter Apps Package)",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to sample (0.0-1.0, default: 1.0)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of samples to process (applied before subset filtering)",
    )

    # --- analyze-traces parameters ---
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path for analysis results (default: security_metrics.json in results_dir)",
    )
    parser.add_argument(
        "--use_gpt",
        action="store_true",
        help="Use GPT for CWE coverage analysis (analyze-traces mode)",
    )
    parser.add_argument(
        "--gpt_model",
        type=str,
        default="gpt-4o-mini",
        help="GPT model for analysis (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=64,
        help="Max concurrent workers for trace analysis (default: 64)",
    )

    # --- generate-dataset parameters ---
    parser.add_argument(
        "--gen_input",
        type=str,
        default=None,
        help="Input filtered dataset path (default: data/dataset_security/KodCode-KodCode-V1_security_relevant.json)",
    )
    parser.add_argument(
        "--gen_output",
        type=str,
        default=None,
        help="Output directory for generated dataset (default: data/dataset_generated/<model>[_<prompt>]/)",
    )
    parser.add_argument(
        "--gen_max_samples",
        type=int,
        default=None,
        help="Randomly sample at most N samples from the input dataset (generate-dataset mode)",
    )
    parser.add_argument(
        "--gen_prompt",
        type=str,
        default="none",
        choices=["none", "direct", "secure", "secure+", "cwe"],
        help="Security prompt to append: none/direct (baseline), secure, secure+ (CWE reasoning), cwe (per-sample CWE hints). Default: none",
    )

    main(parser.parse_args())


if __name__ == "__main__":
    _main()
