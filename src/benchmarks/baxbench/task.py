"""BaxBench benchmark task.

Imports directly from the baxbench codebase (copied locally) to ensure
identical behavior for generation, testing, and evaluation. This includes
baxbench's own Prompter (which handles provider selection, system prompts,
retry logic, batching), Parser (code extraction from LLM responses), and
the full Docker-based testing and evaluation pipeline.

Prompt type mapping (our unified names -> baxbench safety_prompt):
  - direct / none  -> none
  - secure         -> generic
  - secure+        -> secureplus
  - cwe / oracle   -> specific
"""

import pathlib
import sys
from typing import List

from ..base import Benchmark

# The baxbench source is now copied locally into this directory
BAXBENCH_SRC = pathlib.Path(__file__).resolve().parent

# Add local baxbench source to sys.path so its internal imports resolve
if str(BAXBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(BAXBENCH_SRC))

# Map our unified prompt names to baxbench safety_prompt values
PROMPT_MAP = {
    "direct": "none",
    "none": "none",
    "secure": "generic",
    "secure+": "secureplus",
    "secureplus": "secureplus",
    "cwe": "specific",
    "oracle": "specific",
    "specific": "specific",
    # pass-through for native baxbench names
    "generic": "generic",
}


class BaxBenchTask(Benchmark):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ppt = cfg.prompt
        self.bax_safety_prompt = PROMPT_MAP.get(cfg.prompt, cfg.prompt)
        self.n = cfg.n_samples
        self.result_dir = cfg.results_dir / "BaxBench"
        self.spec_type = getattr(cfg, "spec_type", "openapi")
        self.model_name = cfg.model
        self.vllm_flag = getattr(cfg, "vllm", False)
        self.vllm_port = getattr(cfg, "vllm_port", 8000)
        self.openrouter_flag = getattr(cfg, "openrouter", False)
        self.reasoning_effort = getattr(cfg, "reasoning_effort", "high")
        self.scenario_ids = getattr(cfg, "scenarios", None)
        self.env_ids = getattr(cfg, "envs", None)
        self.suffix = getattr(cfg, "suffix", "")
        self.force = getattr(cfg, "force", False)
        self.timeout = getattr(cfg, "timeout", 300)

    def metric(self):
        return "pass@1 / secure_pass@1"

    def __str__(self):
        return "BaxBench"

    def _get_scenarios_and_envs(self):
        """Import and filter scenarios/envs from baxbench."""
        from scenarios import all_scenarios
        from env import all_envs

        scenarios = all_scenarios
        if self.scenario_ids:
            scenarios = [s for s in all_scenarios if s.id in self.scenario_ids]
        scenarios = sorted(scenarios, key=lambda s: s.id)

        envs = all_envs
        if self.env_ids:
            envs = [e for e in all_envs if e.id in self.env_ids]
        envs = sorted(envs, key=lambda e: e.id)

        return scenarios, envs

    def _build_tasks(self):
        """Build baxbench Task objects for all (model, scenario, env) combos."""
        from tasks import Task as BaxTask

        scenarios, envs = self._get_scenarios_and_envs()
        tasks = sorted(
            [
                BaxTask(
                    env=env,
                    scenario=scenario,
                    model=self.model_name,
                    temperature=self.temperature,
                    spec_type=self.spec_type,
                    safety_prompt=self.bax_safety_prompt,
                    reasoning_effort=self.reasoning_effort,
                    openrouter=self.openrouter_flag,
                    vllm=self.vllm_flag,
                    suffix=self.suffix,
                )
                for env in envs
                for scenario in scenarios
            ],
            key=lambda t: t.id,
        )
        return tasks

    def _build_task_handler(self):
        """Build baxbench TaskHandler."""
        from tasks import TaskHandler

        tasks = self._build_tasks()
        return TaskHandler(
            tasks=tasks,
            results_dir=self.result_dir,
            max_concurrent_runs=self.parallel,
        )

    def generate(self, models):
        """Run generation using baxbench's own Prompter and Parser.

        This uses baxbench's Prompter which handles:
        - Provider selection (OpenAI/Anthropic/Together/OpenRouter/vLLM)
          based on model name and flags
        - System prompt: "You are an experienced full-stack developer"
        - Prompt construction via Scenario.build_prompt()
        - Exponential backoff retries
        - Batching (n completions per API call where supported)
        - Response parsing via Parser (code extraction, multi-file support)
        - Trace saving (prompt + response JSON)
        """
        handler = self._build_task_handler()
        handler.run_generation(
            batch_size=self.n,
            max_retries=20,
            base_delay=1.0,
            max_delay=128.0,
            force=self.force,
            openrouter=self.openrouter_flag,
            vllm=self.vllm_flag,
            vllm_port=self.vllm_port,
        )

    def test(self, models):
        """Run Docker-based testing using baxbench's test pipeline.

        For each sample:
        1. Build Docker image with generated code
        2. Run functional tests (stop on first failure)
        3. If all functional tests pass, run security tests
        4. Save test_results.json with CWE findings
        """
        handler = self._build_task_handler()
        samples = list(range(self.n))
        handler.run_tests(
            samples=samples,
            timeout=self.timeout,
            num_ports=10000,
            min_port=12345,
            force=self.force,
        )

    def evaluate(self, ks=None):
        """Evaluate generated code by reading test results and computing metrics.

        Returns a list of (Task, SampleTestResult) tuples with computed metrics
        including pass@k, secure_pass@k, insec_pass, and CWE breakdowns.
        """
        if ks is None:
            ks = [1, 5]
        handler = self._build_task_handler()
        samples = list(range(self.n))
        results = handler.evaluate_results(ks=ks, samples=samples)

        totals = {"n_tasks": 0, "n_ft_correct": 0, "n_ft_and_st_correct": 0,
                  "n_ft_correct_st_incorrect": 0, "n_samples": 0}
        for task, sample_result in results:
            totals["n_tasks"] += 1
            totals["n_samples"] += sample_result.n_samples
            totals["n_ft_correct"] += sample_result.n_ft_correct
            totals["n_ft_and_st_correct"] += sample_result.n_ft_and_st_correct
            totals["n_ft_correct_st_incorrect"] += sample_result.n_ft_correct_st_incorrect

        n = totals["n_tasks"]
        if n > 0:
            totals["pass@1"] = totals["n_ft_correct"] / totals["n_samples"] if totals["n_samples"] > 0 else 0.0
            totals["secure_pass@1"] = totals["n_ft_and_st_correct"] / totals["n_samples"] if totals["n_samples"] > 0 else 0.0
            totals["insec_pass"] = (
                totals["n_ft_correct_st_incorrect"] / totals["n_ft_correct"]
                if totals["n_ft_correct"] > 0 else 0.0
            )

        return results, totals

    def summary(self):
        """Compute and print pass@k metrics using baxbench's evaluation.

        Metrics: pass@k, secure_pass@k, insec_pass, CWE percentages.
        """
        from print import tasks_and_results_to_table, tasks_and_results_to_table_averages

        handler = self._build_task_handler()
        samples = list(range(self.n))
        r = handler.evaluate_results(
            ks=[1, 5],
            samples=samples,
        )
        print(tasks_and_results_to_table_averages(r))
        print()
        print(tasks_and_results_to_table(r, verbose=False))
