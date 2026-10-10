import json
from itertools import product
from pathlib import Path
from typing import List

import pandas as pd
from datasets import load_dataset

from ...helpers import pass_at_k
from ...inference_models import BaseModel, Conversation, Response
from ..prompts import make_prompt
from ..base import Benchmark
from .environment import CWEvalEnv


class CWEvalTask(Benchmark):
    EVAL_SETTINGS = {
        "all": lambda _: True,
    }
    EVAL_SUBSETS = {
        "valid": lambda x: "cwe_400_0_test.py" not in x,
        "python": lambda x: "/py/" in x and "cwe_400_0_test.py" not in x,
        "c": lambda x: "/c/" in x,
        "no_python": lambda x: "/py/" not in x,
        "core_no_python": lambda x: "/py/" not in x and "core" in x,
    }

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ppt = cfg.prompt
        self.prompt_t = make_prompt(cfg.prompt)
        self.n = cfg.n_samples
        self.result_dir = cfg.results_dir / "CWEval"
        self.version = cfg.version if cfg.version is not None else "release_latest"
        self.passat = 1
        self.dataset_size = 119

        self.tag_template = f"{{}}@{self.ppt}@{self.n}@{self.temperature}"

    def metric(self):
        return f"Pass@{self.passat}"

    def __str__(self):
        return "CWEval"

    def get_dataset(self):
        if self.dataset is None:
            self.dataset = load_dataset("moogician/SecCodeGen")["train"]
        return self.dataset

    def generate(self, models: List[BaseModel]):
        dataset = self.get_dataset()
        for model in models:
            tag = self.tag_template.format(str(model))
            out_dir = self.result_dir / tag

            def pre_hook(info):
                line, i = info
                path = out_dir / (
                    f"generated_{i}"
                    + line["task_file_path"][9:].replace("_task", "_raw")
                )
                return path.exists()

            def after_hook(resp, info):
                line, i = info
                path = out_dir / (
                    f"generated_{i}"
                    + line["task_file_path"][9:].replace("_task", "_raw")
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w") as f:
                    f.write(resp.text)

            infos = list(product(dataset, range(self.n)))
            conversations = [
                Conversation(
                    responses=[
                        Response(
                            role="user",
                            text=self.prompt_t.apply(
                                code_prompt=line["code_prompt"],
                                lang=line["lang"],
                                metadata=line,
                            ),
                        )
                    ],
                    system_prompt=None,
                )
                for line, _ in infos
            ]

            model.batch_generate(
                conversations=conversations,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_workers=self.parallel,
                infos=infos,
                pre_hook=pre_hook,
                after_hook=after_hook,
            )

    def test(self, models: List[BaseModel]):
        for tag_dir in self.result_dir.iterdir():
            if not (tag_dir / "res_all.json").exists():
                print(str(tag_dir))
                env = CWEvalEnv()
                try:
                    _ = env.eval(tag_dir)
                    env.collect(tag_dir, tag_dir.parent)
                finally:
                    env.shutdown()

    def summary(self):
        results = []
        for eval_dir in self.result_dir.iterdir():
            if not (eval_dir / "res_all.json").exists():
                print(f"No res_all.json in {eval_dir.name}")
                continue
            model_name, prompt, N, temp = eval_dir.name.split("@")[1:]
            max_token = 24000
            if not self.eval_setting((model_name, prompt, N, temp)):
                continue

            temp, K, N = float(temp), int(self.passat), int(N)
            with open(eval_dir / "res_all.json") as f:
                dddd = json.load(f)
                res_all = {
                    key: val
                    for key, val in filter(
                        lambda x: self.eval_subset(x[0]), dddd.items()
                    )
                }
            data_size = len(
                [
                    line
                    for line in self.get_dataset()
                    if self.eval_subset(line["task_file_path"])
                ]
            )
            functional_rate = (
                sum([sum(x["functional"]) / 5 for x in res_all.values()]) / data_size
            )
            func_sec_rate = (
                sum([sum(x["func_secure"]) / 5 for x in res_all.values()]) / data_size
            )

            identities = {
                "model": model_name,
                "ppt": prompt,
                "temp": temp,
                "n": N,
                "cutoff": max_token,
            }

            results.append(
                {
                    **identities,
                    "func": round(functional_rate * 100, 1),
                    "func-sec": round(func_sec_rate * 100, 1),
                    "func-sec/func": round(
                        func_sec_rate / functional_rate * 100, 1
                    ),
                }
            )

        df = pd.DataFrame(results)
        df = df.sort_values(by="temp")
        df = df.sort_values(by="ppt")
        df = df.sort_values(by="model")
        df = df.sort_values(by="func-sec", ascending=False)
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.min_rows", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_columns", None)
        print(df.to_string())
