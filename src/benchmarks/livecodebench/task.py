import json
import copy
from itertools import product
from pathlib import Path
from typing import List

import pandas as pd
from datasets import load_dataset

from ...helpers import extract_code
from ...inference_models import BaseModel, Conversation, Response
from ..prompts import build_lcb_prompt
from ..base import Benchmark
from .environment import LCBEnv


class LiveCodeBenchTask(Benchmark):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ppt = cfg.prompt
        self.n = cfg.n_samples
        self.result_dir = cfg.results_dir / "LiveCodeBench"
        self.version = cfg.version if cfg.version is not None else "release_latest"
        self.passat = 1
        self.tag_template = f"{{}}@{self.ppt}@{self.n}@{self.temperature}"
        self.version = "v5_v6"

    def metric(self):
        return f"Pass@{self.passat}"

    def __str__(self):
        return "LiveCodeBench"

    def get_dataset(self):
        if self.dataset is None:
            self.dataset = list(
                load_dataset(
                    "livecodebench/code_generation_lite",
                    split="test",
                    version_tag=self.version,
                    trust_remote_code=True,
                )
            )
        return self.dataset

    def generate(self, models: List[BaseModel]):
        dataset = self.get_dataset()
        for model in models:
            tag = self.tag_template.format(str(model))
            out_dir = self.result_dir / tag

            def pre_hook(info):
                line, i = info
                path = out_dir / f"generated_{i}/{line['question_id']}_raw.py"
                return path.exists()

            def after_hook(resp, info):
                line, i = info
                path = out_dir / f"generated_{i}/{line['question_id']}_raw.py"
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w") as f:
                    f.write(resp.text)

            infos = list(product(dataset, range(self.n)))
            conversations = [
                Conversation(
                    responses=[
                        Response(
                            role="user",
                            text=build_lcb_prompt(line, self.ppt),
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
        dataset = self.get_dataset()
        for tag_dir in self.result_dir.iterdir():
            if not (tag_dir / "all_result.json").exists():
                try:
                    self.n = len(
                        [
                            f
                            for f in tag_dir.iterdir()
                            if f.name.startswith("generated_")
                        ]
                    )
                    assert all(
                        [tag_dir / f"generated_{i}" for i in range(self.n)]
                    ), f"{tag_dir} should contain all {self.n} generated folders"
                    for i in range(self.n):
                        if not all(
                            [
                                (
                                    tag_dir
                                    / f"generated_{i}"
                                    / f"{line['question_id']}_raw.py"
                                ).exists()
                                for line in dataset
                            ]
                        ):
                            for line in dataset:
                                if not (
                                    tag_dir
                                    / f"generated_{i}"
                                    / f"{line['question_id']}_raw.py"
                                ).exists():
                                    assert False, (
                                        f"{i}-th generated dir in {tag_dir} should "
                                        f"contain all {len(dataset)} generated raw "
                                        f"responses, {line['question_id']} not found"
                                    )

                    all_result = []
                    for line in dataset:
                        res_line = copy.deepcopy(line)
                        output_list = []
                        code_list = []
                        for i in range(self.n):
                            with open(
                                tag_dir
                                / f"generated_{i}"
                                / f"{line['question_id']}_raw.py"
                            ) as f:
                                output = f.read()
                            output_list.append(output)
                            code_list.append(extract_code(output))
                        res_line["code_list"] = code_list
                        res_line["output_list"] = output_list
                        all_result.append(res_line)
                    with open(tag_dir / "all_result.json", "w") as f:
                        json.dump(all_result, f, indent=4)
                except Exception as e:
                    print(e)
                    continue

            if not (tag_dir / "result.json").exists():
                env = LCBEnv()
                try:
                    _ = env.eval(tag_dir)
                    env.collect(tag_dir)
                finally:
                    env.shutdown()

    def summary(self):
        self.version = "v5_v6"
        ds = self.get_dataset()
        ids = set(line["question_id"] for line in ds)
        results = []
        assert self.passat == 1, "LiveCodeBench only supports pass@1 for now"
        for eval_dir in self.result_dir.iterdir():
            if not (eval_dir / "result.json").exists():
                print(f"No result.json in {eval_dir.name}")
                continue

            model_name, prompt, N, temp = eval_dir.name.split("@")[1:]
            max_token = 24000
            if not self.eval_setting((model_name, prompt, N, temp)):
                continue

            temp, K, N = float(temp), int(self.passat), int(N)
            cnt = 0
            tot = 0
            with open(eval_dir / "all_result.json") as f:
                for line in json.load(f):
                    if line["question_id"] in ids:
                        cnt += line["pass@1"]
                        tot += 1
            try:
                pass_rate = cnt / tot
            except:
                continue

            all_sizes = [
                f.stat().st_size
                for f in sum(
                    [
                        list((eval_dir / f"generated_{i}").iterdir())
                        for i in range(N)
                    ],
                    [],
                )
            ]
            avg_size = sum(all_sizes) / len(all_sizes)

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
                    "pass@1": round(pass_rate * 100, 1),
                    "avg_size": avg_size,
                }
            )
        df = pd.DataFrame(results)
        df = df.sort_values(by="temp")
        df = df.sort_values(by="ppt")
        df = df.sort_values(by="model")
        df = df.sort_values(by="pass@1", ascending=False)
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.min_rows", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_columns", None)
        print(df.to_string())
