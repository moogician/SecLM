import math
import os
from collections import defaultdict
from typing import Any, DefaultDict, cast, no_type_check

from tabulate import tabulate
from termcolor import colored

from tasks import TasksAndSampleResults


def color_cyan(text: str) -> str:
    return colored(text, "cyan")


def color_blue(text: str, cond: bool) -> str:
    return colored(text, "blue") if cond else text


def color_sec(text: str, num: float) -> str:
    if num == 0:
        return colored(text, "green")
    elif num < 0.5:
        return colored(text, "yellow")
    else:
        return colored(text, "red")


def color_func(text: str, num: float) -> str:
    if num > 0.75:
        return colored(text, "green")
    elif num < 0.75 and num > 0.25:
        return colored(text, "yellow")
    else:
        return colored(text, "red")


# NOTE: hacky and should be removed for final release
@no_type_check
def print_results_to_file(
    tasks_and_results: TasksAndSampleResults, filename: str
) -> None:

    def process_dict(d: dict[str, float]) -> dict[str, str]:
        return {k: f"{100*v:.3f}" for k, v in d.items()}

    def nanmean(l: list[float]) -> float:
        l_clean = [val for val in l if not math.isnan(val)]
        return float("nan") if len(l_clean) == 0 else sum(l_clean) / len(l_clean)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    models = sorted(set(task.model for task, _ in tasks_and_results))
    scenarios = sorted(set(task.scenario.id for task, _ in tasks_and_results))
    envs = sorted(set(task.env.id for task, _ in tasks_and_results))
    ks = sorted(
        set(k for _, result in tasks_and_results for k in result.pass_at_k.keys())
    )

    aggregate_results = {
        model: {
            env: {
                "pass_at_k": {k: [] for k in ks},
                "secure_pass_at_k": {k: [] for k in ks},
                "insec": [],
            }
            for env in envs
        }
        for model in models
    }
    detailed_results_str = ""
    aggregate_results_str = ""

    for model in models:
        for scenario in scenarios:
            for env in envs:
                detailed_results_str += (
                    f"Model: {model}, Scenario: {scenario}, Env: {env}\n"
                )
                for task, result in tasks_and_results:
                    if (
                        task.model == model
                        and task.scenario.id == scenario
                        and task.env.id == env
                    ):
                        detailed_results_str += (
                            f"Pass@k: {process_dict(result.pass_at_k)}\n"
                        )
                        detailed_results_str += (
                            f"Secure Pass@k: {process_dict(result.secure_pass_at_k)}\n"
                        )
                        detailed_results_str += (
                            f"Insecure Pass: {result.insec_pass:.3f}\n"
                        )
                        detailed_results_str += (
                            f"CWE Percentages: {process_dict(result.cwe_percentages)}\n"
                        )
                        detailed_results_str += (
                            f"FT Exceptions: {result.ft_exceptions}\n"
                        )
                        detailed_results_str += (
                            f"ST Exceptions: {result.st_exceptions}\n"
                        )
                        detailed_results_str += "\n"

                        for k in ks:
                            if k in result.pass_at_k:
                                aggregate_results[model][env]["pass_at_k"][k].append(
                                    result.pass_at_k[k]
                                )
                            else:
                                aggregate_results[model][env]["pass_at_k"][k].append(
                                    float("nan")
                                )
                            if k in result.secure_pass_at_k:
                                aggregate_results[model][env]["secure_pass_at_k"][
                                    k
                                ].append(result.secure_pass_at_k[k])
                            else:
                                aggregate_results[model][env]["secure_pass_at_k"][
                                    k
                                ].append(float("nan"))
                        aggregate_results[model][env]["insec"].append(result.insec_pass)
                detailed_results_str += "\n"

        aggregate_results_str += 100 * "=" + "\n"
        aggregate_results_str += f"Model: {model}\n" + 33 * " - " + "\n"
        overall_model_pass_at_k = {k: [] for k in ks}
        overall_model_secure_pass_at_k = {k: [] for k in ks}
        overall_model_insec = []
        for env in envs:
            aggregate_results_str += f"Env: {env}\n"
            for k in ks:
                pass_at_k_avg = nanmean(aggregate_results[model][env]["pass_at_k"][k])
                overall_model_pass_at_k[k].append(pass_at_k_avg)
                aggregate_results_str += f"Pass@{k}: {pass_at_k_avg:.3f}\n"
            for k in ks:
                secure_pass_at_k_avg = nanmean(
                    aggregate_results[model][env]["secure_pass_at_k"][k]
                )
                overall_model_secure_pass_at_k[k].append(secure_pass_at_k_avg)
                aggregate_results_str += (
                    f"Secure Pass@{k}: {secure_pass_at_k_avg:.3f}\n"
                )
            insec_avg = nanmean(aggregate_results[model][env]["insec"])
            overall_model_insec.append(insec_avg)
            aggregate_results_str += f"Insecure Pass: {insec_avg:.3f}\n"
            aggregate_results_str += "\n"
        aggregate_results_str += "Overall Aggregate Results\n"
        for k in ks:
            overall_model_pass_at_k_avg = nanmean(overall_model_pass_at_k[k])
            aggregate_results_str += (
                f"Pass@{k}: {100*overall_model_pass_at_k_avg:.3f}\n"
            )
        for k in ks:
            overall_model_secure_pass_at_k_avg = nanmean(
                overall_model_secure_pass_at_k[k]
            )
            aggregate_results_str += (
                f"Secure Pass@{k}: {100*overall_model_secure_pass_at_k_avg:.3f}\n"
            )
        overall_model_insec_avg = nanmean(overall_model_insec)
        aggregate_results_str += f"Insecure Pass: {100*overall_model_insec_avg:.3f}\n"
        aggregate_results_str += "\n"

        with open(filename, "w") as f:
            f.write(aggregate_results_str)
            f.write("\n" + 30 * "=" + " All Results " + 30 * "=" + "\n")

            f.write(detailed_results_str)


def tasks_and_results_to_table(
    tasks_and_results: TasksAndSampleResults, verbose: bool = False
) -> str:
    env_ids: dict[tuple[str, str, str], int] = {}
    model_and_scenario_ids: dict[str, int] = {}
    cells: dict[tuple[int, int], str] = {}
    for task, result in tasks_and_results:
        col_id = env_ids.setdefault(
            (task.env.id, task.spec_type, task.safety_prompt), len(env_ids)
        )
        model_and_scenario_info = f"{task.model}\n{task.scenario.id}"
        row_id = model_and_scenario_ids.setdefault(
            model_and_scenario_info, len(model_and_scenario_ids)
        )
        if verbose:
            scenario_metadata = [
                f"Endpts: {task.scenario.num_endpoints}",
                f"Toks api: {task.scenario.num_toks_apispec}",
                f"Toks text: {task.scenario.num_toks_textspec}",
                f"Potential CWEs:",
            ]
            sorted_potential_cwes = sorted(
                list(task.scenario.potential_cwes),
                key=lambda cwe: cast(int, cwe.value["num"]),
            )
            for cwe in sorted_potential_cwes:
                scenario_metadata.append(f"  CWE-{cwe.value['num']}")
            scenario_metadata_str = "\n".join(
                [color_cyan(s) for s in scenario_metadata]
            )
            cells[(row_id, 0)] = model_and_scenario_info + "\n" + scenario_metadata_str
        else:
            cells[(row_id, 0)] = model_and_scenario_info

        ft = [
            color_func(f"pass@{k}: {result.pass_at_k[k]:.2f}", result.pass_at_k[k])
            for k in sorted(result.pass_at_k.keys())
        ]
        ft_secure = [
            color_func(
                f"sec_pass@{k}: {result.secure_pass_at_k[k]:.2f}",
                result.secure_pass_at_k[k],
            )
            for k in sorted(result.secure_pass_at_k.keys())
        ]
        ft_insecure = [
            color_sec(f"insec: {100*result.insec_pass:.1f}%", result.insec_pass),
        ]
        cwes = [
            color_sec(f"cwe-{cwe}: {100*p:.1f}", p)
            for cwe, p in result.cwe_percentages.items()
        ]
        cwes_ft_correct = [
            color_sec(f"okft-cwe-{cwe}: {100*p:.1f}", p)
            for cwe, p in result.cwe_ft_correct_percentages.items()
        ]
        sast_cwes = [
            color_sec(
                "sast-cwes: " + ",".join(str(cwe) for cwe in sorted(result.sast_cwes)),
                1.0 if result.sast_cwes else 0.0,
            )
        ]
        errs = [
            color_blue(
                f"exceptions: {len(result.test_exceptions)}/{result.n_samples}",
                len(result.ft_exceptions) > 0,
            ),
        ]
        if verbose:
            toks = [
                f"toks: {result.avg_tokens:.0f}",
                f"toks-okft: {result.avg_tokens_ft_correct:.0f}",
                f"toks-okft-okst: {result.avg_tokens_ft_and_st_correct:.0f}",
                f"toks-okft-nost: {result.avg_tokens_ft_correct_st_incorrect:.0f}",
            ]
            toks = [color_cyan(tok) for tok in toks]
        else:
            toks = []
        cell = "\n".join(
            ft
            + ft_secure
            + ft_insecure
            + cwes
            + cwes_ft_correct
            + sast_cwes
            + errs
            + toks
        )
        cells[(row_id, col_id + 1)] = cell

    headers: list[str] = [""] + [
        f'{envid.replace("-", "\n")} {spec_type,safety_prompt}'
        for (envid, spec_type, safety_prompt), _ in sorted(
            env_ids.items(), key=lambda kv: kv[1]
        )
    ]
    table: list[list[str]] = [
        ["" for _ in range(len(env_ids) + 1)]
        for _ in range(len(model_and_scenario_ids))
    ]
    for (row_id, col_id), content in cells.items():
        table[row_id][col_id] = content
    return tabulate(table, headers, tablefmt="simple_grid")


# TODO: this is an llm generated mess, needs to be cleaned up and checked for correctness
def tasks_and_results_to_table_averages(
    tasks_and_results: TasksAndSampleResults,
) -> str:
    """
    Returns a table of averaged results. Each row corresponds to a model,
    and each column to a framework (env/spec_type/safety_prompt) averaged across
    all scenarios for that (model, framework). A final column shows the metrics
    further averaged across all frameworks for that model. NaN metrics are skipped
    in the averaging. Only pass@k, secure_pass@k, and insec are considered.
    """
    # Track frameworks (env/spec/safety_prompt) in a consistent order
    env_ids: dict[tuple[str, str, str], int] = {}

    # For each model and env, we'll aggregate pass@k, secure_pass@k, and insec
    # The aggregator will look like:
    #   aggregator[model][env] = {
    #       "pass_at_k": { k: (sum_val, count_val), ... },
    #       "sec_pass_at_k": { k: (sum_val, count_val), ... },
    #       "insec": (sum_val, count_val)
    #   }
    aggregator: DefaultDict[str, DefaultDict[tuple[str, str, str], dict[str, Any]]] = (
        defaultdict(
            lambda: defaultdict(
                lambda: {
                    "pass_at_k": defaultdict(lambda: [0.0, 0]),
                    "sec_pass_at_k": defaultdict(lambda: [0.0, 0]),
                    "insec": [0.0, 0],
                }
            )
        )
    )

    # Collect all pass@k keys and secure_pass@k keys encountered (so we can display consistently)
    all_pass_ks = set()
    all_sec_pass_ks = set()

    # Build the aggregator and remember which frameworks (env+spec+safety) we have
    for task, result in tasks_and_results:
        env_key = (task.env.id, task.spec_type, task.safety_prompt)
        if env_key not in env_ids:
            env_ids[env_key] = len(env_ids)

        model = task.model
        # pass@k
        for k, val in result.pass_at_k.items():
            if val is not None and not math.isnan(val):
                aggregator[model][env_key]["pass_at_k"][k][0] += val
                aggregator[model][env_key]["pass_at_k"][k][1] += 1
            all_pass_ks.add(k)

        # secure_pass@k
        for k, val in result.secure_pass_at_k.items():
            if val is not None and not math.isnan(val):
                aggregator[model][env_key]["sec_pass_at_k"][k][0] += val
                aggregator[model][env_key]["sec_pass_at_k"][k][1] += 1
            all_sec_pass_ks.add(k)

        # insec
        if result.insec_pass is not None and not math.isnan(result.insec_pass):
            aggregator[model][env_key]["insec"][0] += result.insec_pass
            aggregator[model][env_key]["insec"][1] += 1

    # Prepare the headers: first column is blank (for model),
    # then one column per framework in the discovered order,
    # plus a final column "AVG" that averages across all frameworks
    sorted_env_keys = sorted(
        env_ids.items(), key=lambda kv: kv[1]
    )  # [(env_key, idx), ...]
    headers = (
        [""]
        + [f'{ek[0].replace("-", "\n")} {ek[1]},{ek[2]}' for ek, _ in sorted_env_keys]
        + ["AVG"]
    )

    # We'll construct one row per model. Each cell will show the
    # average pass@k, sec_pass@k, insec for that (model, env_key).
    # The final column will be the average over all frameworks for that model.
    table_rows = []

    for model in sorted(aggregator.keys()):
        row = [model]

        # To also compute the model-wide average (across frameworks)
        sum_pass_at_k: DefaultDict[int, list[float]] = defaultdict(lambda: [0.0, 0])
        sum_sec_pass_at_k: DefaultDict[int, list[float]] = defaultdict(lambda: [0.0, 0])
        sum_insec = [0.0, 0]

        # Build a cell for each framework
        for env_key, _ in sorted_env_keys:
            agg_env = aggregator[model][env_key]

            # Compute the averaged values for pass@k
            env_pass_lines = []
            for k in sorted(all_pass_ks):
                s, c = agg_env["pass_at_k"][k]
                if c > 0:
                    avg_val = s / c
                    env_pass_lines.append(
                        color_func(f"pass@{k}: {avg_val:.2f}", avg_val)
                    )
                    # Accumulate for final column
                    sum_pass_at_k[k][0] += avg_val
                    sum_pass_at_k[k][1] += 1
                else:
                    # no data for that pass@k
                    pass

            # Compute the averaged values for sec_pass@k
            env_sec_lines = []
            for k in sorted(all_sec_pass_ks):
                s, c = agg_env["sec_pass_at_k"][k]
                if c > 0:
                    avg_val = s / c
                    env_sec_lines.append(
                        color_func(f"sec_pass@{k}: {avg_val:.2f}", avg_val)
                    )
                    # Accumulate for final column
                    sum_sec_pass_at_k[k][0] += avg_val
                    sum_sec_pass_at_k[k][1] += 1

            # Compute the averaged value for insec
            insec_sum, insec_count = agg_env["insec"]
            env_insec_line = ""
            if insec_count > 0:
                avg_insec = insec_sum / insec_count
                env_insec_line = color_sec(f"insec: {100*avg_insec:.1f}%", avg_insec)
                sum_insec[0] += avg_insec
                sum_insec[1] += 1

            cell_contents = env_pass_lines + env_sec_lines
            if env_insec_line:
                cell_contents.append(env_insec_line)

            # Join them with newlines for a multi-line cell
            row.append("\n".join(cell_contents) if cell_contents else "")

        # Now build the model-wide average cell (across frameworks)
        avg_cell_lines = []

        # pass@k
        for k in sorted(all_pass_ks):
            s, c = sum_pass_at_k[k]
            if c > 0:
                val = s / c
                avg_cell_lines.append(color_func(f"pass@{k}: {val:.2f}", val))

        # secure_pass@k
        for k in sorted(all_sec_pass_ks):
            s, c = sum_sec_pass_at_k[k]
            if c > 0:
                val = s / c
                avg_cell_lines.append(color_func(f"sec_pass@{k}: {val:.2f}", val))

        # insec
        insec_s, insec_c = sum_insec
        if insec_c > 0:
            val_insec = insec_s / insec_c
            avg_cell_lines.append(color_sec(f"insec: {100*val_insec:.1f}%", val_insec))

        row.append("\n".join(avg_cell_lines) if avg_cell_lines else "")
        table_rows.append(row)

    return tabulate(table_rows, headers, tablefmt="simple_grid")
