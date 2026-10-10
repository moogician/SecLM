"""
Filter KodCode-V1 dataset for security-relevant samples.

Uses an LLM to analyze each coding instruction + solution for potential
CWE security vulnerabilities. Samples flagged as security-relevant are
saved alongside their vulnerability annotations.

Usage:
  python -m src filter-dataset \
      --checker_model gpt-4o --checker_provider openai \
      --subset Filter Apps Package --sample_fraction 0.1
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

import datasets
import tqdm

from .inference_models import BaseModel, Conversation, Response, get_inference_model

SAVE_PATH = Path("data/dataset_security")

ANALYSIS_SYSTEM_PROMPT = (
    "You are a helpful assistant and a code security expert. "
    "You are especially skilled in recognizing code snippets and coding "
    "scenarios that might result in CWE security vulnerabilities."
)

ANALYSIS_PROMPT_NON_REASONING = """
Please analyze the following coding instruction and code snippet for security implications. First reason step-by-step through the coding instruction and the snippet, thinking of potential security vulnerabilities that could occur in this context. Think also of security vulnerabilities that could occur in different implementations to the same instruction. However, be very conservative in your assessment; for instance, only consider a user-input dependent security vulnerability if it is very clear from the context of the instruction and the code that it is actually processing user input. Then, provide a summary of your analysis, and finally, output the CWE security vulnerabilities you have identified and if the vulnerability of the coding task depends on direct user input or no. Use the following format for your final output:

<VULNERABILITIES>
["CWE-123", "CWE-456", ...]
</VULNERABILITIES>

<USER_INPUT_DEPENDENCY>
True of False
</USER_INPUT_DEPENDENCY>

Here is the coding instruction:
{instruction}

And here is the code snippet fulfilling the instruction:
{code}
"""

ANALYSIS_PROMPT_REASONING = """
Please analyze the following coding instruction and code snippet for security implications. Think of potential security vulnerabilities that could occur in this context. Think also of security vulnerabilities that could occur in different implementations to the same instruction. However, be very conservative in your assessment; for instance, only consider a user-input dependent security vulnerability if it is very clear from the context of the instruction and the code that it is actually processing user input. Then, provide a summary of your analysis, and finally, output the CWE security vulnerabilities you have identified and if the vulnerability of the coding task depends on direct user input or no. Use the following format for your final output:

<VULNERABILITIES>
["CWE-123", "CWE-456", ...]
</VULNERABILITIES>

<USER_INPUT_DEPENDENCY>
True of False
</USER_INPUT_DEPENDENCY>

Here is the coding instruction:
{instruction}

And here is the code snippet fulfilling the instruction:
{code}
"""


def _esc(name: str) -> str:
    return name.replace("/", "-")


def _parse_response(response: str) -> tuple[list[str], bool]:
    """Parse LLM response for vulnerability list and user-input dependency."""
    vulnerabilities_pattern = re.search(
        r"<VULNERABILITIES>(.*?)</VULNERABILITIES>", response, re.DOTALL
    )
    if not vulnerabilities_pattern:
        return [], False

    vulnerabilities_text = vulnerabilities_pattern.group(1).strip()
    try:
        vulnerabilities_text = vulnerabilities_text.replace("'", '"')
        vul_list = json.loads(vulnerabilities_text)
    except json.JSONDecodeError:
        cwe_pattern = re.compile(r'"(CWE-\d+)"')
        vul_list = cwe_pattern.findall(vulnerabilities_text)

    user_input_pattern = re.search(
        r"<USER_INPUT_DEPENDENCY>(.*?)</USER_INPUT_DEPENDENCY>", response, re.DOTALL
    )
    if not user_input_pattern:
        return vul_list, False
    user_input_text = user_input_pattern.group(1).strip()
    user_input_dependency = (
        "true" in user_input_text.lower()
        and "false" not in user_input_text.lower()
    )
    return vul_list, user_input_dependency


def _analyze_example(example: dict, analyzer: BaseModel) -> Tuple[dict, str]:
    """Analyze a single KodCode-V1 example for security vulnerabilities."""
    if analyzer.reasoning:
        prompt_text = ANALYSIS_PROMPT_REASONING.format(
            instruction=example["question"], code=example["solution"]
        )
    else:
        prompt_text = ANALYSIS_PROMPT_NON_REASONING.format(
            instruction=example["question"], code=example["solution"]
        )

    prompt = Conversation(
        system_prompt=ANALYSIS_SYSTEM_PROMPT,
        responses=[Response(role="user", text=prompt_text)],
    )

    response = analyzer.generate_erb(
        prompt, temperature=0.0, max_retries=10, base_delay=1, max_delay=120
    )

    vulnerabilities, user_input_dependent = _parse_response(response.text)
    return {
        "sample_id": example["question_id"],
        "security": len(vulnerabilities) > 0,
        "vulnerabilities": vulnerabilities,
        "user_input_dependent": user_input_dependent,
        "error": None,
        "question": example["question"],
        "solution": example["solution"],
    }, response.text


def filter_dataset(cfg) -> None:
    """Filter KodCode-V1 for security-relevant samples."""
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    dataset_name = "KodCode/KodCode-V1"

    save_file_relevant = SAVE_PATH / f"{_esc(dataset_name)}_security_relevant.json"
    if save_file_relevant.exists() and not cfg.force:
        print(f"Results already exist at {save_file_relevant}. Use --force to overwrite.")
        return

    # Load dataset
    print(f"Loading {dataset_name}...")
    dataset = datasets.load_dataset(dataset_name)["train"]

    # Filter by subset if specified
    if cfg.subset:
        dataset = dataset.filter(lambda x: x["subset"] in cfg.subset)
        print(f"Filtered to subsets {cfg.subset}: {len(dataset)} samples")

    # Sub-sample if requested
    if cfg.sample_fraction < 1.0:
        n = max(1, int(len(dataset) * cfg.sample_fraction))
        dataset = dataset.select(range(n))
        print(f"Sub-sampled to {cfg.sample_fraction:.0%}: {len(dataset)} samples")

    # Apply absolute cap
    if cfg.max_samples is not None and cfg.max_samples < len(dataset):
        dataset = dataset.select(range(cfg.max_samples))
        print(f"Capped to {cfg.max_samples} samples")

    print(f"Analyzing {len(dataset)} samples for security relevance...")

    # Build the checker model
    model = get_inference_model(
        model_name=cfg.checker_model,
        model_provider=cfg.checker_provider,
        reasoning=cfg.checker_reasoning,
        reasoning_effort=cfg.checker_reasoning_effort,
    )

    # Save response directory
    resp_dir = SAVE_PATH / "resps"
    resp_dir.mkdir(parents=True, exist_ok=True)

    # Analyze concurrently
    with tqdm.tqdm(total=len(dataset)) as pbar:
        pbar.get_lock()

        def analyze_with_progress(example: dict) -> Tuple[dict, str]:
            try:
                result, response = _analyze_example(example, model)
            except Exception as e:
                result = {
                    "sample_id": example["question_id"],
                    "security": False,
                    "vulnerabilities": [],
                    "user_input_dependent": False,
                    "error": str(e),
                    "question": example["question"],
                    "solution": example["solution"],
                }
                response = "ERRORED"
                print(f"Error on {example['question_id']}: {e}")
            with pbar.get_lock():
                pbar.update(1)
            return result, response

        with ThreadPoolExecutor(max_workers=cfg.max_concurrent_runs) as executor:
            pairs = list(executor.map(analyze_with_progress, dataset))

    # Save individual responses
    for result, response in pairs:
        result["analysis_model"] = model.model_name
        with open(resp_dir / f"{result['sample_id']}.resp", "w") as f:
            f.write(response)

    # Split into security-relevant and security-irrelevant
    results = [pair[0] for pair in pairs]
    security_relevant = [r for r in results if r["security"]]
    security_irrelevant = [r for r in results if not r["security"]]

    save_file_relevant = SAVE_PATH / f"{_esc(dataset_name)}_security_relevant.json"
    save_file_irrelevant = SAVE_PATH / f"{_esc(dataset_name)}_security_irrelevant.json"

    with open(save_file_relevant, "w") as f:
        json.dump(security_relevant, f, indent=4)
    with open(save_file_irrelevant, "w") as f:
        json.dump(security_irrelevant, f, indent=4)

    # Print summary
    n_errors = sum(1 for r in results if r["error"] is not None)
    print(f"\nDone.")
    print(f"  Total: {len(results)}")
    print(f"  Security-relevant: {len(security_relevant)} -> {save_file_relevant}")
    print(f"  Security-irrelevant: {len(security_irrelevant)} -> {save_file_irrelevant}")
    print(f"  Errors: {n_errors}")
