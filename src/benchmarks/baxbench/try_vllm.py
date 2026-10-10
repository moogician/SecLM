import httpx
import json, sys
from openai import OpenAI
from pathlib import Path
from together import Together
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger

from prompts import Parser
from env import env_dict
prefill_template = sys.argv[1]
assert prefill_template in ['prefill1', "none"]

MODEL = "lifted-79-prefill1-correct"
PORT = 27979

long_timeout_client = httpx.Client(timeout=18000)
client = OpenAI(
            base_url=f"http://localhost:{PORT}/v1",
            api_key="EMPTY",
            http_client=long_timeout_client,
        )

logger = getLogger("prefill")

def contruct_prompt(line):
    if "question" in line:
        question1, question2 = line['question'].split("Make sure that the code is runnable and that it starts a webserver on port 0.0.0.0:5000.")
        question1 = question1 + "Make sure that the code is runnable and that it starts a webserver on port 0.0.0.0:5000."
        if "Provide the source code in the following format:" in question2:
            question=question1+"\n\nProvide the source code in the following format:" \
                + question2.split("Provide the source code in the following format:")[1]
        else:
            question=question1
        return question
    elif len(line) == 2:
        assert 'role' in line[0] and 'content' in line[0]
        return line[0]['content']
    else:
        raise ValueError("format not supported")

with open("cwes.json") as f:
    all_cwes = json.load(f)

#cwe_entry = all_cwes[str(int(cwe_id))]
#vulnerabilities.append(f"- CWE-{cwe_entry['ID']}: {cwe_entry['name']}" + f" ({cwe_entry['alt_terms'][0]['name']})" if 'alt_terms' in cwe_entry else '')


tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B")
ds = [line for line in load_dataset("eth-sri/baxbench")['train'] if line['scenario'] in ('ProfileCollection', 'ImageTransfer', 'RecommendationService', 'Forum', 'Recipes')]
prompts = []
#prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#print(prompt)

prefill1_prefix_template = "Important considerations:\n 1. We must avoid common vulnerabilities ("

for line in ds:
    line = dict(line)
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": line['none_prompt']}],
                                           tokenize=False, add_generation_prompt=True)
    #vulnerabilities.append(f"- CWE-{cwe_entry['ID']}: {cwe_entry['name']}" + f" ({cwe_entry['alt_terms'][0]['name']})" if 'alt_terms' in cwe_entry      else '')
    if prefill_template == "prefill1":
        prefill_prefix = prefill1_prefix_template
    else:
        raise ValueError(prefill_template)
    prompt += prefill_prefix
    prompts.append(prompt)
    
input(prompts[0])


# Print output

with tqdm(total=len(prompts)) as pbar:

    def gen_sample(prompt):
        response = client.completions.create(
            model=MODEL,
            prompt=prompt,
            max_tokens=10000,
            temperature=0.6,
            top_p=0.9,
        ).choices[0].text.strip() # type: ignore
        with pbar.get_lock():
            pbar.update(1)
        return response
    with ThreadPoolExecutor(max_workers=32) as executor:
       resps = list(executor.map(
                   gen_sample,
                   prompts)
               )

for line, prompt, resp in zip(ds, prompts, resps):
    line = dict(line)
    fmw = '/'.join(line['env'].split('-')[1:])
    env = env_dict[fmw]
    
    files = Parser(env, logger).parse_response(resp)
    try:
        # save code
        "results/deepseek-ai-DeepSeek-R1/Apps_10050_C/Go-Fiber/temp0.6-openapi-securec/sample0/code"
        code_dir = Path("results-prefill") / MODEL / line['scenario'] / line['env'] \
                / f'temp0.6-openapi-{prefill_template}' / f"sample0" / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        for path, code in files.items():
            full_path = code_dir / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w") as f:
                f.write(code)
        # save trace
        with open(code_dir.parent / "trace.log", "w") as f:
            f.write(prompt + "\n\n" + '='*20 + '\n\n' + resp)

    except Exception as e:
        pass
    
