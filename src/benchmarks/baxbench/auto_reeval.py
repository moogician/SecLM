import subprocess, traceback, json, time
import sys
import os
from contextlib import contextmanager
from pathlib import Path
from datasets import get_dataset_split_names, load_dataset 
from logging import getLogger
import logging

from prompts import Parser
from env import env_dict

original_stdout = sys.stdout

@contextmanager
def suppress_stdout():
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

logging.disable(logging.CRITICAL + 1)
logger = getLogger("auto_eval")

def parse_split_name(split_name: str):
    parsed = split_name.split('.')
    model = "@".join(parsed[:-2])
    temp = float(parsed[-2].replace("_", "."))
    prompt_style = parsed[-1]
    return model, temp, prompt_style

def list_splits(dataset_name, config_name=None):
    splits = get_dataset_split_names(dataset_name, config_name=config_name)
    return splits

def download_trace(dataset_name, split):
    model, temp, prompt_style = parse_split_name(split)
    ds = [dict(line) for line in load_dataset(dataset_name, split=split)] # type: ignore
    #dataset = {"scenario":[], "env":[], "sample": [], "trace":[], "prompt": []}
    for line in ds:
        env = line['env']

        trace = line['trace']
        with open(f"results/{model}/{line['scenario']}/{env}/temp{temp}-openapi-{prompt_style}/sample{line['sample']}/trace.txt", "w") as f:
            f.write(trace)

    # WARNING: NO INTEGRITY CHECK
    assert len(ds) % 392 == 0

def download_results(dataset_name, split) -> int:
    model, temp, prompt_style = parse_split_name(split)
    ds = [dict(line) for line in load_dataset(dataset_name, split=split)] # type: ignore
    return len(ds) // 392

with suppress_stdout():
    splits = list_splits("eth-sri/baxbench-results-FINAL")

# reeval all splits 
for split in splits:
    model, temp, prompt_style = parse_split_name(split)
    print(model, temp, prompt_style)
    # print(f"Downloading {model}, {temp}, {prompt_style}")
    with suppress_stdout():
        n_samples = download_results("eth-sri/baxbench-results-FINAL", split)
    # print(f"Finished Downloading {model}, {temp}, {prompt_style}")

    with suppress_stdout():
        subprocess.run(['pipenv', 'run', 'python', 'src/main.py', 
                        "--models", model, 
                        '--mode', 'test', 
                        '--scenarios', 'PDFToText', 
                        '--n_samples', str(n_samples), 
                        '--temperature', str(temp),
                        '--spec_type', 'openapi',
                        '--safety_prompt', prompt_style,
                        '--max_concurrent_runs', '32',
                        '--timeout', '60',
                        '--force'])
    # print(f"Finished Evaluating {model}, {temp}, {prompt_style}")
    subprocess.run(['pipenv', 'run', 'python', 'src/calc_stat.py', f'results/{model}', prompt_style, str(n_samples)])
