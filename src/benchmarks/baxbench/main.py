import argparse
import json
import os 
import pathlib
from typing import Any, no_type_check

import docker

import sast
from env import all_envs
from env.base import COMMON_DOCKER_RUN_COMMANDS
from print import (
    print_results_to_file,
    tasks_and_results_to_table,
    tasks_and_results_to_table_averages,
)
from scenarios import all_scenarios
from tasks import Task, TaskHandler, esc

_DEFAULT_SAVE_PATH = pathlib.Path(__file__).parent.parent / "results"


@no_type_check
def strip_string_output(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, str):
            return result.strip()
        return result

    return wrapper


@no_type_check
@strip_string_output
def clean_docker_file(docker_file: str, env_id: str) -> str:
    if env_id.lower().startswith("rust") or env_id.lower().startswith("go"):
        docker_file = docker_file.split("# build the generated code")[0].strip()
        docker_file += "\nENV APP_SECRET=supers3cret"
        return docker_file
    elif env_id.lower().startswith("php"):
        return docker_file.split("ENTRYPOINT")[0]
    elif env_id.lower().startswith("ruby"):
        docker_file = "\n".join(line for line in docker_file.split("\n")[:-2])
        return docker_file
    elif "nest" in env_id.lower():
        docker_file = "\n".join(line for line in docker_file.split("\n")[:-5])
        docker_file += "\nENV APP_SECRET=supers3cret"
        return docker_file
    elif "javascript" in env_id.lower():
        docker_file = "\n".join(line for line in docker_file.split("\n")[:-3])
        docker_file += "\nENV APP_SECRET=supers3cret"
        return docker_file
    elif "python" in env_id.lower():
        docker_file = "\n".join(line for line in docker_file.split("\n")[:-2])
        return docker_file
    else:
        return docker_file


def main(args: Any) -> None:

    # ----- Preparation -----#
    envs = all_envs
    exclude_envs = args.exclude_envs if args.exclude_envs else []
    envs = [e for e in all_envs if e.id not in exclude_envs]
    if args.envs:
        envs = [e for e in all_envs if e.id in args.envs]
    envs = sorted(envs, key=lambda e: e.id)
    if args.mode == "explain":
        # Envs are irrelevant for explain mode.
        envs = [all_envs[0]]

    if not envs:
        raise Exception(
            f"Got an empty/invalid list of envs, possible choices: {[e.id for e in all_envs]}",
        )

    exclude_scenarios = args.exclude_scenarios if args.exclude_scenarios else []
    scenarios = [e for e in all_scenarios if e.id not in exclude_scenarios]
    if args.scenarios:
        scenarios = [
            e
            for e in all_scenarios
            if e.id in args.scenarios and e.id not in exclude_scenarios
        ]
    scenarios = sorted(scenarios, key=lambda s: s.id)
    if not scenarios:
        raise Exception(
            f"Got an empty/invalid list of scenarios, possible choices: {[s.id for s in all_scenarios]}",
        )

    if not args.models:
        raise Exception("Got an empty list of models")

    if args.only_samples:
        samples = args.only_samples
    else:
        samples = list(range(args.n_samples))

    if args.ks:
        ks = args.ks
    else:
        ks = [1, 5]

    tasks = sorted(
        [
            Task(
                env=env,
                scenario=scenario,
                model=model,
                temperature=args.temperature,
                spec_type=args.spec_type,
                safety_prompt=args.safety_prompt,
                reasoning_effort=args.reasoning_effort,
                openrouter=args.openrouter,
                vllm=args.vllm,
                suffix=args.suffix
            )
            for env in envs
            for scenario in scenarios
            for model in args.models
        ],
        key=lambda t: t.id,
    )

    task_handler = TaskHandler(
        tasks=tasks,
        results_dir=args.results_dir,
        max_concurrent_runs=args.max_concurrent_runs,
    )

    # ----- Run tasks -----#

    if args.mode == "generate":
        task_handler.run_generation(
            batch_size=args.n_samples,
            max_retries=args.max_retries,
            base_delay=args.base_delay,
            max_delay=args.max_delay,
            force=args.force,
            openrouter=args.openrouter,
            vllm=args.vllm,
            vllm_port=args.vllm_port,
        )
    elif args.mode == "test":
        task_handler.run_tests(
            samples=samples,
            timeout=args.timeout,
            num_ports=args.num_ports,
            min_port=args.min_port,
            force=args.force,
        )
        if args.prune_docker:
            docker.from_env().containers.prune()
    elif args.mode == "evaluate":
        r = task_handler.evaluate_results(
            ks=ks,
            samples=samples,
        )
        print(tasks_and_results_to_table_averages(r))
        print()
        print(tasks_and_results_to_table(r, verbose=False))

    # NOTE: hack modes, remove for release
    elif args.mode == "print_to_file":
        r = task_handler.evaluate_results(
            ks=ks,
            samples=samples,
        )
        models = "-".join([esc(m) for m in args.models])
        file_name = pathlib.Path(
            f"commit_results/results-{models}-temp{args.temperature}-{args.spec_type}-{args.safety_prompt}-{args.n_samples}.txt"
        )
        print_results_to_file(r, file_name)
    elif args.mode == "snyk_code":
        sast_reports = sast.run_snyk_code(args.results_dir, args.force)
        task_handler.save_sast_reports(sast_reports, samples)
    elif args.mode == "get_prompts":
        print("Building the prompts")
        prompts = task_handler.get_prompts(agent=False)
        file_name = (
            pathlib.Path(__file__).parent.parent
            / "prompts"
            / (args.spec_type + "-" + args.safety_prompt + ".json")
        )
        file_name.parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, "w") as f:
            json.dump(prompts, f, indent=4)

    elif args.mode == "get_agent_artifacts":
        print("Generating the agent artifacts")
        prompts = task_handler.get_prompts(agent=True)
        full_agent_artifacts: dict[str, dict[str, str | dict[str, str] | list[str]]] = (
            {}
        )
        reference_path = pathlib.Path(__file__).parent.parent / "results" / "o3-mini"
        for env in envs:
            for scenario in scenarios:
                sol_path = (
                    reference_path
                    / scenario.id
                    / (env.id).replace("/", "-")
                    / "temp0.0-openapi-none"
                    / "sample0"
                    / "code"
                )
                paths_to_keep = []
                if sol_path.exists():
                    for root, dirs, files in os.walk(sol_path):
                        rel_root = os.path.relpath(root, sol_path)
                        for file in files:
                            rel_path = (
                                os.path.join(rel_root, file)
                                if rel_root != "."
                                else file
                            )
                            paths_to_keep.append(rel_path)

                extra_docker_commands = (
                    COMMON_DOCKER_RUN_COMMANDS
                    + scenario.needed_packages.get("_all_", [])
                    + scenario.needed_packages.get(env.language, [])
                )
                df = env.build_only_docker_image_file(extra_docker_commands)
                df = clean_docker_file(df, env.id)
                mf = env.manifest_files

                full_agent_artifacts[scenario.id + "-" + env.id] = {
                    "dockerfile": df.strip(),
                    "manifest_files": {
                        filename: file.strip() for filename, file in mf.items()
                    },
                    "prompt": prompts[scenario.id + "-" + env.id],
                    "paths_to_keep": paths_to_keep,
                }
        file_name = (
            pathlib.Path(__file__).parent.parent
            / "agent_artifacts"
            / (args.spec_type + "-" + args.safety_prompt + ".json")
        )
        file_name.parent.mkdir(parents=True, exist_ok=True)
        print("Saving the agent artifacts to", file_name)
        with open(file_name, "w") as f:
            json.dump(full_agent_artifacts, f, indent=4)
    else:
        raise Exception(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", type=str, nargs="+", required=True, help="List of models"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "generate",
            "test",
            "evaluate",
            "print_to_file",
            "snyk_code",
            "get_agent_artifacts",
            "get_prompts",
        ],
        required=True,
        help="Mode in which to run the code",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for sampling"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="The number of samples to generate or test. Will index from 0.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="high",
        choices=["low", "medium", "high"],
        help="The reasoning effort to use for reasoning models.",
    )
    parser.add_argument(
        "--only_samples",
        type=int,
        nargs="+",
        default=None,
        help="If given, it will restrict operations to these sample indices.",
    )
    parser.add_argument(
        "--ks", type=int, nargs="+", default=None, help="List of k for pass@k score."
    )
    parser.add_argument(
        "--envs",
        type=str,
        default=None,
        nargs="+",
        help="List of environments (if empty, then all environments are used)",
    )
    parser.add_argument(
        "--exclude_envs",
        type=str,
        default=None,
        nargs="+",
        help="List of environments to exclude",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        nargs="+",
        help="List of scenarios (if empty, then all scenarios are used)",
    )
    parser.add_argument(
        "--exclude_scenarios",
        type=str,
        default=None,
        nargs="+",
        help="List of scenarios to exclude",
    )
    parser.add_argument(
        "--spec_type",
        choices=["openapi", "text"],
        default="openapi",
        type=str,
        help="The type of specifications to use.",
    )
    parser.add_argument(
        "--safety_prompt",
        #choices=["none", "generic", "specific"],
        default="none",
        type=str,
        help="The type of additional safety cue to use.",
    )
    parser.add_argument(
        "--results_dir",
        type=pathlib.Path,
        default=_DEFAULT_SAVE_PATH,
        help="Directory to save the results",
    )
    parser.add_argument(
        "--max_concurrent_runs",
        type=int,
        default=None,
        help="Maximum number of concurrent runs",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for each test run in seconds.",
    )
    parser.add_argument(
        "--num_ports",
        type=int,
        default=10000,
        help="Number of ports to use for docker containers",
    )
    parser.add_argument(
        "--min_port",
        type=int,
        default=12345,
        help="Minimum port number to use for docker containers",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=20,
        help="Maximum number of retries for backoff during generation",
    )
    parser.add_argument(
        "--base_delay",
        type=float,
        default=1.0,
        help="Base delay for backoff during generation",
    )
    parser.add_argument(
        "--max_delay",
        type=float,
        default=128.0,
        help="Maximum delay for backoff during generation",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force generation even if the file already exists",
    )
    parser.add_argument(
        "--prune_docker",
        action="store_true",
        help="Prune docker containers after running tests",
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help="Route requests through OpenRouter",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use VLLM for generation",
    )
    parser.add_argument(
        "--vllm_port",
        type=int,
        default=8000,
        help="Port for VLLM server",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="suffix for result dir",
    )
    main(parser.parse_args())
