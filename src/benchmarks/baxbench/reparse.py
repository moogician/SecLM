import json, sys
import shutil
from pathlib import Path
from together import Together
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger

from prompts import Parser
from env import env_dict
res_folder, sample_count = sys.argv[1:]
sample_count = int(sample_count)
res_folder = Path(res_folder)

ds = [dict(line) for line in load_dataset("eth-sri/baxbench")['train']]

#results/eth-sri-lifted-22-79-agentic3-correct-mix/Wiki/Python-Flask/temp0.6-openapi-none/gen.log
gen_log_dict = {}
for gen_log_path in res_folder.glob("**/gen.log"):
    fmw = gen_log_path.parents[1].name
    scenario = gen_log_path.parents[2].name
    assert ( scenario, fmw) not in gen_log_dict, (scenario, fmw)
    gen_log_dict[(scenario, fmw)] = gen_log_path

def extract_traces(trace_file):
    with open(trace_file) as f:
        raw = f.read()
    raw = "got model responses:".join(raw.split("got model responses:")[1:])
    rets = raw.split('<<<RESPONSE DELIM>>>')
    rets[-1] = 'INFO'.join(('-'*100).join(rets[-1].split('-'*100)[:-1]).split("INFO")[:-1])
    return rets

logger = getLogger("aha")

for line in ds:
    env = line['env']
    fmw = '/'.join(line['env'].split('-')[1:])
    gen_log_path = gen_log_dict[(line['scenario'], line['env'])]
    traces = extract_traces(gen_log_path)
    if (line['scenario'], line['env']) == ("ClickCount", "Python-Django"):
        print("HERE IS BIG")
        input(traces[0])
    assert len(traces) == sample_count, (line['scenario'], env)
    for idx, trace in enumerate(traces):
        print(env, line['scenario'])
        files = Parser(env_dict[fmw], logger).parse_response(trace)
    
        code_dir = gen_log_path.parent / f"sample{idx}" / "code" 
        #assert code_dir.exists()
        if code_dir.exists():
            shutil.rmtree(code_dir)
            code_dir.mkdir()
        else:
            code_dir.mkdir(parents=True)

        for path, code in files.items():
            full_path = code_dir / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w") as f:
                f.write(code)
