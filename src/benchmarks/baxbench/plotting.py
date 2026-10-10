# mypy: ignore-errors
import argparse
import pathlib
import pickle
import typing
from typing import Any, DefaultDict, cast, no_type_check

import docker
from matplotlib import pyplot as plt

from cwes import CWE

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

import sast
from env import Env, all_envs
from scenarios import all_scenarios
from tasks import SampleTestResult, Task, TaskHandler


@typing.no_type_check
def fetch_all():
    envs = all_envs
    envs = sorted(envs, key=lambda e: e.id)

    scenarios = all_scenarios
    scenarios = sorted(scenarios, key=lambda s: s.id)

    all_models = [
        "claude-3-5-sonnet-latest",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3",
        "gpt-4o",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "mistralai/codestral-2501",
        "o1",
        "o3-mini",
        "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
    ]
    models = all_models

    ks = [1, 5]
    spec_type = "openapi"
    reasoning_effort = "high"
    openrouter = False
    _DEFAULT_SAVE_PATH = pathlib.Path(__file__).parent.parent / "results"

    # NON REASONING MODEL TASKS
    tasks = sorted(
        [
            Task(
                env=env,
                scenario=scenario,
                model=model,
                temperature=0.4,
                spec_type=spec_type,
                safety_prompt=safety_prompt,
                reasoning_effort=reasoning_effort,
                openrouter=openrouter,
            )
            for safety_prompt in ["none", "specific", "generic"]
            for env in envs
            for scenario in scenarios
            for model in models
            if model not in ["deepseek-ai/DeepSeek-R1", "o1", "o3-mini"]
        ],
        key=lambda t: t.id,
    )
    task_handler = TaskHandler(
        tasks=tasks, results_dir=_DEFAULT_SAVE_PATH, max_concurrent_runs=None
    )
    r = task_handler.evaluate_results(
        ks=ks,
        samples=list(range(10)),
    )
    # plot_results(r)

    # REASONING MODEL TASKS
    tasks2 = sorted(
        [
            Task(
                env=env,
                scenario=scenario,
                model=model,
                temperature=0.0,
                spec_type=spec_type,
                safety_prompt=safety_prompt,
                reasoning_effort=reasoning_effort,
                openrouter=openrouter,
            )
            for safety_prompt in ["none", "specific", "generic"]
            for env in envs
            for scenario in scenarios
            for model in models
            if model in ["deepseek-ai/DeepSeek-R1", "o1", "o3-mini"]
        ],
        key=lambda t: t.id,
    )
    task_handler2 = TaskHandler(
        tasks=tasks2, results_dir=_DEFAULT_SAVE_PATH, max_concurrent_runs=None
    )
    r2 = task_handler2.evaluate_results(
        ks=ks,
        samples=list(range(1)),
    )
    r.extend(r2)
    print(len(r))
    assert (
        len(r) == 3 * len(all_models) * 28 * 14
    )  # 3 prompts * 10 models * 28 scenarios * 14 envs
    with open("fullresults.pkl", "wb") as f:
        pickle.dump(r, f)
    plot_results(r)


@typing.no_type_check
def better_tabulate(label, caption, headers, rows):
    headers, formats = zip(*headers)
    headers = [f"\\textbf{{{header}}}" for header in headers]
    joined_formats = "\n".join(formats)
    lines = [
        "\\begin{table*}[h] \\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        "\\resizebox{\columnwidth}{!}{",
        f"\\begin{{tabular}}{{\n{joined_formats}\n}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join([str(f) for f in row]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(lines)


@typing.no_type_check
def tabulate_scenarios(tasks_and_results):
    models = sorted(set(task.model for task, _ in tasks_and_results))
    scenarios = sorted(set(task.scenario for task, _ in tasks_and_results))
    envs = sorted(set(task.env.id for task, _ in tasks_and_results))
    out_dir = "./results/tables/"
    # 1: static table: scenario_name, desc, endpoints, lines api, lines text, potential CWEs (everyone has 703)
    headers = [
        ("Scenario", "l"),
        ("Description", "l"),
        ("\#Endpoints", r"x{1}{0}"),
        ("API Spec Tokens", r"x{3}{0}"),
        ("Text Spec Tokens", r"x{3}{0}"),
        ("Potential CWEs", "l"),
    ]
    rows = []
    totaleps, apispec, textspec, numcwes = 0, 0, 0, 0
    allcwes = set()
    for scenario in scenarios:
        sorted_potential_cwes = sorted(
            [
                cwe.value["num"] if cwe.value["num"] != 77 else 78
                for cwe in scenario.potential_cwes
            ]
        )
        row = [
            scenario.id,
            scenario.short_app_description,
            scenario.num_endpoints,
            scenario.num_toks_apispec,
            scenario.num_toks_textspec,
            ", ".join([str(cwe) for cwe in sorted_potential_cwes]),
        ]
        rows.append(row)
        apispec += scenario.num_toks_apispec
        textspec += scenario.num_toks_textspec
        numcwes += len(sorted_potential_cwes)
        allcwes.update(sorted_potential_cwes)
        totaleps += scenario.num_endpoints
    apispec /= len(scenarios)
    textspec /= len(scenarios)
    numcwes /= len(scenarios)
    allcwes = sorted(allcwes)
    caption = f"Summary of our {len(scenarios)} scenarios. We report the number of endpoints (\#EPs, {totaleps} total), the API spec length in GPT4o tokens (average: {apispec:.1f}), the text spec length (average: {textspec:.1f} tokens), and the potential CWEs (on average {numcwes:.1f} per scenario, in total {len(allcwes)} unique ones."

    with open(out_dir + "scenarios_raw.tex", "w") as f:
        f.write(better_tabulate("scenarios", caption, headers, rows))


@typing.no_type_check
def tabulate_cwes(tasks_and_results):
    out_dir = "./results/tables/"
    allcwes = set()
    scenarios = sorted(set(task.scenario for task, _ in tasks_and_results))
    for scenario in scenarios:
        for cwe in scenario.potential_cwes:
            if cwe.value["num"] != 77:
                allcwes.update({cwe})
            else:
                allcwes.update({CWE.OS_INJECTION})
    allcwes = list(sorted(allcwes, key=lambda cwe: cwe.value["num"]))

    headers = [
        ("CWE", "l"),
        ("CWE Description", "l"),
        ("OWASP", "l"),
    ]
    rows = []
    for cwe in allcwes:
        name = cwe.name
        nametoks = cwe.name.split("_")
        # change uppercase to only first letter up
        nametoks = [tok.capitalize() for tok in nametoks]
        name = " ".join(nametoks)
        row = [f"{cwe.value['num']}: {name}", cwe.value["desc"], "OWASP TODO"]
        rows.append(row)
    caption = f"Summary of all {len(allcwes)} CWEs that we cover."

    with open(out_dir + "cwes_raw.tex", "w") as f:
        f.write(better_tabulate("cwes", caption, headers, rows))


@typing.no_type_check
def plot_passatone_vs_toksAPI(basedir, tasks_and_results):
    scenarios = sorted(set(task.scenario for task, _ in tasks_and_results))
    for safety_prompt in ["none", "specific", "generic"]:
        fun = dict()
        funsec = dict()
        it = 0
        for task, result in tasks_and_results:
            if safety_prompt != task.safety_prompt:
                continue
            it += 1
            scenario = task.scenario
            if scenario not in fun:
                fun[scenario] = []
                funsec[scenario] = []
            try:
                fun[scenario].append(result.pass_at_k[1])
                funsec[scenario].append(result.secure_pass_at_k[1])
            except Exception as e:
                print(
                    f"Error for {task.safety_prompt} {task.scenario.id} {task.model} {task.env.id}: {e}\n{result}"
                )
                fun[scenario].append(0.0)
                funsec[scenario].append(0.0)
        print(f"Processed {it} tasks")
        lens = [len(fun[s]) for s in scenarios]
        lens_ = [len(funsec[s]) for s in scenarios]
        lens = [lens[0] == l for l in lens]
        lens_ = [lens_[0] == l for l in lens_]
        assert all(lens), "Not all scenarios have the same number of fun results"
        assert all(lens_), "Not all scenarios have the same number of funcsec results"
        for s_ in fun.keys():
            print(f"{s_.id}: {len(fun[s_])}")
            print(f"{s_.id}: {len(funsec[s_])}")

        for dic, name, colors in [
            (fun, "fun", ("firebrick", "lightsalmon")),
            (funsec, "fun+sec", ("steelblue", "skyblue")),
        ]:
            if name == "fun+sec":
                continue
            apispecs = [s.num_toks_apispec for s in scenarios]
            avgfuns = [sum(dic[s]) / len(dic[s]) for s in scenarios]
            min_scen = None
            min_avg = None
            for avg, scen in zip(avgfuns, scenarios):
                if min_avg is None or avg < min_avg:
                    min_avg = avg
                    min_scen = scen
            print(
                "MINIMUM FUNCK IN SCEN: ",
                min_scen.id,
                min_avg,
                min_scen.num_toks_apispec,
            )
            import numpy as np
            from numpy.polynomial.polynomial import Polynomial

            plt.clf()
            plt.figure(figsize=(10, 8))
            p = Polynomial.fit(apispecs, avgfuns, 1)
            corr = np.around(np.corrcoef(apispecs, avgfuns), 2)[0, 1]
            plt.plot(
                *p.linspace(),
                color=colors[1],
                linewidth=5,
                linestyle="--",
                alpha=0.8,
                label=f"OLS fit (" + r"$\rho$" + f"={corr})",
                zorder=1,
            )
            plt.scatter(
                apispecs,
                avgfuns,
                color=colors[1],
                s=300,
                edgecolors=colors[0],
                linewidths=3,
                alpha=0.8,
                zorder=2,
            )

            plt.tick_params(axis="x", which="both", bottom=False, top=False)
            plt.tick_params(axis="y", which="both", left=False, right=False)
            plt.xlabel("#Tokens in OpenAPI Specification", fontsize=20, labelpad=15)
            plt.ylabel("pass@1", fontsize=20, labelpad=15)
            plt.yticks(
                [0.1 * (i + 1) for i in range(6)],
                [str(10 * (i + 1)) + "%" for i in range(6)],
                fontsize=20,
            )
            plt.xticks([200 + 100 * i for i in range(8)], fontsize=20)
            plt.xlim(120, 980)
            plt.ylim(0.05, 0.65)
            plt.legend(fontsize=20)
            # plt.title(f"Pass@1 vs. API Spec Tokens {name} (prompt: {safety_prompt})")

            # linear regression plot and add to scatter
            # plt.show()
            plt.box(False)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{basedir}/{safety_prompt}_pass1_vs_api_{name}.pdf")
            # plt.clf()
            # ttextspecs = [s.num_toks_textspec for s in scenarios]
            # avgfuns = [sum(dic[s]) / len(dic[s]) for s in scenarios]
            # plt.scatter(ttextspecs, avgfuns)
            # plt.legend()
            # plt.title(f"Pass@1 vs. Text Spec Tokens {name}")
            # plt.show()
            # plt.savefig(f"{basedir}/pass1_vs_text_{name}.png")


@typing.no_type_check
def plot_programlen_ratios(basedir, tasks_and_results):
    models = sorted(set(task.model for task, _ in tasks_and_results))
    scenarios = sorted(set(task.scenario.id for task, _ in tasks_and_results))
    envs = sorted(set(task.env.id for task, _ in tasks_and_results))
    print(models, scenarios, envs)

    ratios = dict()
    for task, result in tasks_and_results:
        scenario = task.scenario
        if (
            result.n_tokens_ft_and_st_correct > 0
            and result.avg_tokens_ft_correct_st_incorrect > 0
        ):
            ratio = (
                result.avg_tokens_ft_and_st_correct
                / result.avg_tokens_ft_correct_st_incorrect
            )
            tag = (
                f"{task.env.id}::{task.scenario.id}::{task.model}::{task.safety_prompt}"
            )
            if tag in ratios:
                raise RuntimeError(f"Duplicate tag: {tag}")
            ratios[tag] = ratio

    avg_global = (
        sum(ratios.values()) / len(ratios.values()) - 1.0
    )  ###############################

    print(f"{len(ratios)} ratios computed from {len(tasks_and_results)} tasks total")
    # Slice by each
    for i, nameslabel in enumerate(
        [(envs, "env"), (scenarios, "scenario"), (models, "model")]
    ):
        print("Plotting by {}".format(nameslabel[1]))
        names, label = nameslabel
        names = set(names)
        ratios_grouped = {name: [] for name in names}
        for tag, ratio in ratios.items():
            tagtoks = tag.split("::")
            assert len(tagtoks) == 4
            if tagtoks[i] in names:
                ratios_grouped[tagtoks[i]].append(ratio)

        # Plot
        plt.clf()
        plt.figure(figsize=(20, 5))
        data = []
        chek = 0
        chekn = 0
        for name in names:
            if len(ratios_grouped[name]) > 0:
                avg = sum(ratios_grouped[name]) / len(ratios_grouped[name]) - 1.0
                shortname = (
                    name.replace("JavaScript", "JS")
                    .replace("Python", "Py")
                    .replace("Ruby", "Rb")[:8]
                    + "."
                )
                chek += avg * len(ratios_grouped[name])
                chekn += len(ratios_grouped[name])
                if label == "scenario":
                    shortname = shortname[:5]
                shortname = f"{shortname} \n ({len(ratios_grouped[name])})"
                data.append((shortname, avg))
            else:
                print(f"No data for {name} when grouping by {label}")
        print(f"{avg_global} should be == to {chek / chekn}")
        data = reversed(sorted(data, key=lambda x: x[1]))
        names_nonzero, ratios_grouped_avg = zip(*data)

        print(30 * "-")
        print(
            "Avg %increase of tokens in FT+ST-passing vs. FT-passing, grouped by {label} (both prompts)"
        )
        print(
            "\n".join(
                [f"{n}: {100*r:.1f}" for n, r in zip(names_nonzero, ratios_grouped_avg)]
            )
        )
        plt.bar(names_nonzero, ratios_grouped_avg)
        plt.hlines(0, -1, len(names_nonzero), colors="r", linestyles="dashed")
        plt.title(
            f"Avg %increase of tokens in FT+ST-passing vs. FT-passing, grouped by {label} (both prompts) (global avg: {avg_global:0.4f})"
        )
        plt.savefig(f"{basedir}/ratios_{label}.pdf")


@typing.no_type_check
def tabulate_cwe_occurence(basedir, tasks_and_results):
    models = sorted(set(task.model for task, _ in tasks_and_results))
    scenarios = sorted(set(task.scenario.id for task, _ in tasks_and_results))
    envs = sorted(set(task.env.id for task, _ in tasks_and_results))
    print(models, scenarios, envs)

    allcwes = set()
    for task, _ in tasks_and_results:
        allcwes.update(
            [
                cwe.value["num"]
                for cwe in task.scenario.potential_cwes
                if cwe.value["num"] != 77
            ]
        )
    print(sorted(allcwes), len(allcwes))
    for task, result in tasks_and_results:
        cwes_here = [
            cwe.value["num"] if cwe.value["num"] != 77 else 78
            for cwe in result.cwes.keys()
        ]
        for cwe in cwes_here:
            if cwe not in allcwes:
                raise RuntimeError(f"Unknown CWE {cwe} shows up in results")

    for safety_prompt in ["none", "specific", "generic"]:
        cwecounts = dict()
        it = 0
        for task, result in tasks_and_results:
            if task.safety_prompt != safety_prompt:  # only none!
                continue
            it += 1
            curr_cwecounts = {cwe: 0 for cwe in allcwes}
            curr_cwecounts["total"] = 0
            curr_cwecounts["total_couldcwe"] = {}
            for cwe in allcwes:
                curr_cwecounts["total_couldcwe"][cwe] = 0
            if result.n_ft_correct > 0:
                curr_cwecounts["total"] = result.n_ft_correct
                for potential_cwe in task.scenario.potential_cwes:
                    if potential_cwe.value["num"] == 77:
                        potential_cwe_num = 78
                    else:
                        potential_cwe_num = potential_cwe.value["num"]
                    curr_cwecounts[f"total_couldcwe"][
                        potential_cwe_num
                    ] = result.n_ft_correct

                for cwe, cnt in result.cwes_ft_correct.items():
                    key = cwe.value["num"] if cwe.value["num"] != 77 else 78
                    curr_cwecounts[key] = cnt

                tag = f"{task.env.id}::{task.scenario.id}::{task.model}"
                if tag in cwecounts:
                    raise RuntimeError(f"Duplicate tag: {tag}")
                cwecounts[tag] = curr_cwecounts
        print(f"{len(cwecounts)} cwecountdicts computed from {it} tasks total")

        # Slice by each
        for i, nameslabel in enumerate(
            [(envs, "env"), (scenarios, "scenario"), (models, "model")]
        ):
            headers = [(nameslabel[1], "l")]
            for cwe in sorted(allcwes):
                headers.append((cwe, "x{1}{2}"))
            rows = []
            print("Plotting by {}".format(nameslabel[1]))
            names, label = nameslabel
            names = set(names)

            cwecounts_grouped = {}
            for tag, curr_cwecounts in cwecounts.items():
                tagtoks = tag.split("::")
                assert len(tagtoks) == 3
                if tagtoks[i] not in names:
                    raise RuntimeError(f"Unknown {label} {tagtoks[i]}")
                name = tagtoks[i]
                if name not in cwecounts_grouped:
                    cwecounts_grouped[name] = {}
                    cwecounts_grouped[name]["total_couldcwe"] = {}
                    for cwe in allcwes:
                        cwecounts_grouped[name][cwe] = 0
                        cwecounts_grouped[name]["total_couldcwe"][cwe] = 0
                    cwecounts_grouped[name]["total"] = 0

                cwecounts_grouped[tagtoks[i]]["total"] += curr_cwecounts["total"]
                for cwe in allcwes:
                    cwecounts_grouped[tagtoks[i]]["total_couldcwe"][
                        cwe
                    ] += curr_cwecounts["total_couldcwe"][cwe]
                    cwecounts_grouped[tagtoks[i]][cwe] += curr_cwecounts[cwe]

            # Plot
            # plt.clf()
            # plt.figure(figsize=(20, 5))
            data = []
            for name in sorted(names):
                row = [name]
                # plt.clf()
                if name not in cwecounts_grouped:
                    print(f"No data for {name} when grouping by {label}")
                    for cwe in allcwes:
                        row.append("N/A")
                    continue
                # print(f"For {name} there is {cwecounts_grouped[name]['total']} FT-passing code")
                # print(f"For {name} there is {cwecounts_grouped[name]['total_couldcwe'][703]} FT-passing code that COULD have cwe 703")

                cweratio_pairs = []
                for cwe in sorted(allcwes):
                    if cwecounts_grouped[name]["total_couldcwe"][cwe] == 0:
                        row.append("/")
                    else:
                        rat = (
                            cwecounts_grouped[name][cwe]
                            / cwecounts_grouped[name]["total_couldcwe"][cwe]
                        )
                        if rat > 0.9:
                            row.append(f"\\textcolor{{brown}}{{{rat:0.2f}}}")
                        else:
                            row.append(f"{rat:0.2f}")
                    cweratio_pairs.append((cwe, rat))
                cweratio_pairs = sorted(
                    cweratio_pairs, key=lambda x: x[1], reverse=True
                )
                # bar plot
                # plt.bar(
                #    [str(cwe) for cwe, _ in cweratio_pairs],
                #    [ratio for _, ratio in cweratio_pairs],
                # )
                # plt.title(f"Occurence of CWEs in FT-passing code, for {label} (safety prompt={safety_prompt})")
                # plt.savefig(f"{basedir}cwes/{safety_prompt}_{label}_{name.replace("/", "-")}.png")
                rows.append(row)

            # Also tabulate
            with open(
                f"{basedir}../tables/{safety_prompt}_cwes_{label}_raw.tex", "w"
            ) as f:
                rows = better_tabulate(
                    f"cwes_{label}",
                    f"For each CWE: Percent of FT-passing code that has it / Percent of FT-passing code that could have it (per {label}) (safetyprompt = {safety_prompt})",
                    headers,
                    rows,
                )
                f.write(rows)


# Analyzes the results from r and plots them to out_dir
# Task: env, scenario, model, id
# SampleTestResult: contrary to its name, its aggregated over samples
@typing.no_type_check
def plot_results(tasks_and_results: list[tuple[Task, SampleTestResult]]) -> None:
    # tabulate_scenarios(tasks_and_results) # DONE, manually tuned
    # tabulate_cwes(tasks_and_results) # DONE, manually tuned

    basedir = "./results/plots/scatterplots/"
    # plot_passatone_vs_toksAPI(basedir, tasks_and_results)
    basedir = "./results/plots/ratios/"
    # plot_programlen_ratios(basedir, tasks_and_results)

    basedir = "./results/tables/"
    tabulate_cwe_occurence(basedir, tasks_and_results)


# main
if __name__ == "__main__":
    try:
        with open("fullresults.pkl", "rb") as f:
            r = pickle.load(f)
    except FileNotFoundError:
        r = fetch_all()
    plot_results(r)
