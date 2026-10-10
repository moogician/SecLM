import json, sys
import traceback
from transformers import AutoTokenizer # type: ignore
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger

from vllm import LLM, SamplingParams

from prompts import Parser
from env import env_dict
model, temp, prompt_style, sample_time = sys.argv[1:]
escaped_model = model.replace("/", "@")
temp = float(temp)
sample_time = int(sample_time)
assert prompt_style in ['none']

logger = getLogger("prefill")

ds = [dict(line) for line in load_dataset("eth-sri/baxbench")['train']]
prompts = []

for line in ds:
    line = dict(line)
    code_dir = Path("results") / escaped_model / line['scenario'] / line['env'] \
            / f'temp{temp}-openapi-{prompt_style}' / f"sample0" / "code"
    if code_dir.exists():
        raise ValueError("At least part of the dataset has been generated!")

    assert prompt_style == "none"
    prompts.append(line['none_prompt'])
    
#input(prompts[0])

tokenizer = AutoTokenizer.from_pretrained(model)
client = LLM(model=model, trust_remote_code=True, max_model_len=32768, tensor_parallel_size=4)

params = SamplingParams(temperature=temp, n=sample_time, seed=114514, max_tokens=24000)
resps = client.generate([tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False,
                            add_generation_prompt=True
                            ) for prompt in prompts], 
                        sampling_params=params)
rets = [[resp.outputs[idx].text for idx in range(sample_time)] for resp in resps]

for line, prompt, resps in zip(ds, prompts, rets):
    line = dict(line)
    fmw = '/'.join(line['env'].split('-')[1:])
    env = env_dict[fmw]
    for idx, resp in enumerate(resps):
        files = Parser(env, logger).parse_response(resp)

        code_dir = Path("results") / escaped_model / line['scenario'] / line['env'] \
            / f'temp{temp}-openapi-{prompt_style}' / f"sample{idx}" / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
	# save trace
        with open(code_dir.parent / "trace.log", "w") as f:
            f.write(prompt + "\n\n" + '='*20 + '\n\n' + resp)
        with open(code_dir.parent / "trace.json", "w") as f:
            json.dump({'prompt': prompt, 'resp': resp}, f)

        try:
            # save code
            for path, code in files.items():
                full_path = code_dir / path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(code)
        except Exception as e:
            traceback.print_exc()
