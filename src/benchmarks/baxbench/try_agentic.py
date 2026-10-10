import json, sys, re
from pathlib import Path
from together import Together
from openai import OpenAI
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger

from prompts import Parser
from env import env_dict
agentic_template = sys.argv[1]
assert agentic_template in ['vanilla1', 'rewrite1']

#CWEPREDICT_PROMPT = "Here is an OpenAPI specification for a webserver:\n ```\n{api_spec}\n```\n\nHowever, there are security risks in this task. Please analyze this problem and respond with potential CWEs that are in this scenario."

CWEPREDICT_PROMPT = "Here is an OpenAPI specification for a webserver:\n ```\n{api_spec}\n```\n\nHowever, there are security risks in this task. Specifically,\n{cwe_descs}\n\nPlease analyze this problem and reason about how these CWEs are related."


# Create client
client = Together()  # or use env var
openai_client = OpenAI()

logger = getLogger("agentic_prefill")

def contruct_prompt(question):
    question1, question2 = question.split("Make sure that the code is runnable and that it starts a webserver on port 0.0.0.0:5000.")
    question1 = question1 + "Make sure that the code is runnable and that it starts a webserver on port 0.0.0.0:5000.\nYour code should be secure and should NOT contain any vulnerabilities.\nYou should first identify possible CWEs from the given instruction, choose the most probable ONE, write  code that is functional, and then make sure that the code does not contain the CWEs you choose while keeping it functional.\nYou should choose the MOST ELEGANT implementation."
    if "Provide the source code in the following format:" in question2:
        question=question1+"\n\nProvide the source code in the following format:" \
            + question2.split("Provide the source code in the following format:")[1]
    else:
        question=question1
    return question

def extract_int(text):
    # Find the first integer in the string
    match = re.search(r'\d+', text)

    return int(match.group()) if match else None

def extract_cwe_codes(text):
    pattern = r'CWE-\d+'
    return re.findall(pattern, text, re.IGNORECASE)


def query_chat(c, model, prompts, _filter = lambda x: True):
    with tqdm(total=len(prompts)) as pbar:

        def gen_sample(prompt):
            while True:
                response = c.chat.completions.create(
                    model=model,
                    messages=prompt,
                    max_tokens=36000,
                    temperature=0.6,
                    top_p=0.9,
                ).choices[0].message.content.strip() # type: ignore
                if _filter(response):
                    break
            with pbar.get_lock():
                pbar.update(1)
            return response
        with ThreadPoolExecutor(max_workers=32) as executor:
           return list(executor.map(
                       gen_sample,
                       [[{"content": prompt, "role": "user"}] for prompt in prompts])
                   )

def query_completion(c, model, prompts, _filter = lambda x: True):
    with tqdm(total=len(prompts)) as pbar:

        def gen_sample(prompt):
            while True:
                response = c.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=36000,
                    temperature=0.6,
                    top_p=0.9,
                ).choices[0].text.strip() # type: ignore
                if _filter(response):
                    break
            with pbar.get_lock():
                pbar.update(1)
            return response
        with ThreadPoolExecutor(max_workers=32) as executor:
           return list(executor.map(gen_sample, prompts))



with open("cwes.json") as f:
    all_cwes = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
ds = [dict(line) for line in load_dataset("eth-sri/baxbench")['train'] if line['scenario'] in ['RecommendationService','Forum','ImageTransfer','ProfileCollection', 'Recipes'] + ['Compiler','FileSearch','FrameExtract','ImageConverter', 'ImageTransfer', 'PDFCat', 'PDFToText', 'RegexSearch', 'SongDownloader', 'ZipToTxt']][:1]

prompts = []

specs_to_check = set([(line['api_spec'], line['scenario'], tuple(line['cwes'])) for line in ds])
assert len(specs_to_check) == len(set([line['scenario'] for line in ds])), len(specs_to_check)

def get_cwepred_dir(scenario):
    return Path("cwe-preds") / scenario

remaining_specs = []
prompts = []
for spec, scenario, cwe_ids in specs_to_check:
    if get_cwepred_dir(scenario).exists():
        continue
    cwes = ', '.join([f'CWE-{idx}' for idx in cwe_ids])
    #vulnerabilities.append(f"- CWE-{cwe_entry['ID']}: {cwe_entry['name']}" + f" ({cwe_entry['alt_terms'][0]['name']})" if 'alt_terms' in cwe_entry      else '')
    cwe_descs = '\n'.join([f"- CWE-{cwe_entry['ID']}: {cwe_entry['name'].replace("'", '')}" + 
                           (f" ({cwe_entry['alt_terms'][0]['name']})" if 'alt_terms' in cwe_entry and '(' not in cwe_entry['name'] else '')
                           for cwe_entry in [all_cwes[str(cwe)] for cwe in cwe_ids]])

    remaining_specs.append((spec, scenario))
    #prompts.append(CWEPREDICT_PROMPT.format(api_spec=spec))
    prompts.append(CWEPREDICT_PROMPT.format(api_spec=spec, cwe_descs=cwe_descs))

resps = query_chat(client, "deepseek-ai/DeepSeek-R1", prompts)

for prompt, resp, (spec, scenario) in zip(prompts, resps, remaining_specs):
    (get_cwepred_dir(scenario) / "sample0").mkdir(parents=True, exist_ok=True)
    data = {"scenario": scenario, "resp": resp, "cwes": extract_cwe_codes(resp.split("</think>")[1])}
    with open(get_cwepred_dir(scenario) / "sample0" / "log.txt", "w") as f:
        f.write(prompt + "\n\n" + "="*20 + "\n\n" + resp)

    with open(get_cwepred_dir(scenario) / "sample0" / "log.json", "w") as f:
        json.dump(data, f)



# load log
scenario_to_log = {line['scenario']: [] for line in ds}
for line in ds:
    scenario = line['scenario']
    (get_cwepred_dir(scenario) / "sample0").mkdir(parents=True, exist_ok=True)

    with open(get_cwepred_dir(scenario) / "sample0" / "log.json") as f:
        log = json.load(f)
    scenario_to_log[scenario].append(log)

#for line in ds:
#    scenario = line['scenario']
#    cwe_ids = [f"CWE-{cwe}" for cwe in line['cwes'] if cwe != 703]

    #for log in scenario_to_log[scenario]:
        #assert all(cwe_id in log['cwes'] for cwe_id in cwe_ids), f"{cwe_ids}\n{log['cwes']}\n{scenario}"




