"""
Generate a reasoning+solution dataset from the security-filtered KodCode samples.

Reads the filtered dataset (produced by filter-dataset), sends each question to
an LLM (optionally with a security prompt appended), and saves the generated
reasoning and solution alongside the original metadata.

Usage:
  # Basic generation (no security prompt)
  python -m src --mode generate-dataset \
      --model deepseek-ai/DeepSeek-R1 --provider vllm

  # With a security prompt appended
  python -m src --mode generate-dataset \
      --model gpt-4o --provider openai --gen_prompt secure+

  # Custom input/output paths
  python -m src --mode generate-dataset \
      --model gpt-4o --provider openai \
      --gen_input data/my_filtered.json \
      --gen_output data/my_output/
"""

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

import tqdm

from .inference_models import BaseModel, Conversation, Response, get_inference_model

DEFAULT_INPUT = Path("data/dataset_security/KodCode-KodCode-V1_security_relevant.json")
DEFAULT_OUTPUT_DIR = Path("data/dataset_generated")

SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Write correct, high-quality code that solves the given problem."
)

BASE_USER_PROMPT = """\
Solve the following coding problem. Think step-by-step, reason about the approach, and then provide a complete Python solution.

{question}

Provide your final solution inside a single code block wrapped by triple backticks."""

# Security prompt snippets that can be appended (keyed by --gen_prompt value).
# "cwe" is built dynamically per-sample using its vulnerabilities list.
SECURITY_PROMPTS = {
    "none": "",
    "direct": "",
    "secure": (
        "\n\nYour code should be secure and should NOT contain any vulnerabilities."
    ),
    "secure+": (
        "\n\nYour code should be secure and should NOT contain any vulnerabilities. "
        "You should first identify possible CWEs from the given instruction, write "
        "code that is functional, and then make sure that the code does not contain "
        "any CWEs you identified while keeping it functional."
    ),
}


def _cwe_suffix(vulnerabilities: list[str]) -> str:
    """Build a per-sample CWE-specific security prompt from the vulnerabilities list."""
    base = (
        "\n\nYour code should be secure and should NOT contain any vulnerabilities."
    )
    if vulnerabilities:
        cwe_list = ", ".join(vulnerabilities)
        base += f" Specifically, avoid the following vulnerabilities: {cwe_list}."
    return base


def _esc(name: str) -> str:
    return name.replace("/", "@")


def _build_prompt(
    question: str, prompt_type: str, vulnerabilities: list[str] | None = None
) -> str:
    """Build the user prompt, optionally appending a security reminder."""
    base = BASE_USER_PROMPT.format(question=question)
    if prompt_type == "cwe":
        return base + _cwe_suffix(vulnerabilities or [])
    suffix = SECURITY_PROMPTS.get(prompt_type)
    if suffix is None:
        raise ValueError(
            f"Unknown prompt type: {prompt_type}. "
            f"Available: {', '.join(list(SECURITY_PROMPTS.keys()) + ['cwe'])}"
        )
    return base + suffix


def _generate_one(
    sample: dict, model: BaseModel, prompt_type: str
) -> Tuple[dict, str]:
    """Generate reasoning + solution for a single sample."""
    user_text = _build_prompt(
        sample["question"], prompt_type, sample.get("vulnerabilities")
    )
    conv = Conversation(
        system_prompt=SYSTEM_PROMPT,
        responses=[Response(role="user", text=user_text)],
    )

    response = model.generate_erb(
        conv, temperature=0.6, max_retries=10, base_delay=1, max_delay=120
    )

    result = {
        **sample,
        "generation_model": model.model_name,
        "generation_prompt": prompt_type,
        "generated_reasoning": response.reasoning,
        "generated_solution": response.text,
        "generation_error": None,
    }
    return result, response.text


def generate_dataset(cfg) -> None:
    """Generate a reasoning+solution dataset from filtered security samples."""
    # Determine input path
    input_path = Path(getattr(cfg, "gen_input", None) or DEFAULT_INPUT)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found at {input_path}. "
            "Run --mode filter-dataset first, or specify --gen_input."
        )

    # Load filtered dataset
    print(f"Loading filtered dataset from {input_path}...")
    with open(input_path) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples")

    # Determine prompt type
    prompt_type = getattr(cfg, "gen_prompt", None) or "none"

    # Determine output directory
    output_dir = getattr(cfg, "gen_output", None)
    if output_dir:
        output_dir = Path(output_dir)
    else:
        # Default name includes model and prompt type
        dir_name = _esc(cfg.model)
        if prompt_type not in ("none", "direct"):
            dir_name += f"_{prompt_type}"
        output_dir = DEFAULT_OUTPUT_DIR / dir_name

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "generated.json"
    resp_dir = output_dir / "resps"
    resp_dir.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not cfg.force:
        print(f"Results already exist at {output_file}. Use --force to overwrite.")
        return

    # Build model
    model = get_inference_model(
        model_name=cfg.model,
        model_provider=cfg.provider,
        reasoning=getattr(cfg, "checker_reasoning", False),
        reasoning_effort=getattr(cfg, "checker_reasoning_effort", None),
    )

    # Apply gen_max_samples with random sampling if specified
    gen_max = getattr(cfg, "gen_max_samples", None)
    if gen_max is not None and gen_max < len(samples):
        total = len(samples)
        random.seed(42)
        samples = random.sample(samples, gen_max)
        print(f"Randomly sampled {gen_max} from {total} samples")

    print(
        f"Generating with model={model.model_name}, "
        f"prompt={prompt_type}, output={output_dir}"
    )

    # Generate concurrently
    with tqdm.tqdm(total=len(samples)) as pbar:
        pbar.get_lock()

        def gen_with_progress(sample: dict) -> Tuple[dict, str]:
            try:
                result, raw = _generate_one(sample, model, prompt_type)
            except Exception as e:
                result = {
                    **sample,
                    "generation_model": model.model_name,
                    "generation_prompt": prompt_type,
                    "generated_reasoning": "",
                    "generated_solution": "",
                    "generation_error": str(e),
                }
                raw = "ERRORED"
                print(f"Error on {sample['sample_id']}: {e}")
            with pbar.get_lock():
                pbar.update(1)
            return result, raw

        with ThreadPoolExecutor(max_workers=cfg.max_concurrent_runs) as executor:
            pairs = list(executor.map(gen_with_progress, samples))

    # Save individual responses
    for result, raw in pairs:
        with open(resp_dir / f"{result['sample_id']}.resp", "w") as f:
            f.write(raw)

    # Save aggregated results
    results = [pair[0] for pair in pairs]
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    # Summary
    n_errors = sum(1 for r in results if r["generation_error"] is not None)
    print(f"\nDone. Results saved to {output_file}")
    print(f"  Total: {len(results)}")
    print(f"  Errors: {n_errors}")
