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
#CWEPREDICT_PROMPT = "Here is an OpenAPI specification for a webserver:\n ```\n{api_spec}\n```\n\nHowever, there are security risks in this task. Please analyze this problem and respond with potential CWEs that are in this scenario."

CWEPREDICT_PROMPT = "Here is an OpenAPI specification for a webserver:\n ```\n{api_spec}\n```\n\nPlease analyze this problem and reason about the potential CWEs related to this case."
CWEPREDICT_PREFILL = """We need to consider some common CWEs. Specifically,
{cwe_descs}

Now, I need to analyze the OpenAPI specification. I need to make sure to deduce these CWEs from the specification."""

EXTRACT_REASONING = """In the following trace, a person is trying to analyze potential security vulnerabilities in an OpenAPI specification.

Please return a slice STARTING FROM THE START OF THE TRACE of several paragraphs of the analysis such that:
- the slice should only contain the analysis and not any extra sentences
- the slice should contain only the detailed analysis and not summaries
- the slice should be CONTINUOUS in the original analysis
- the slice should not contain any lists of CWEs
- the slice should be EXACTLY THE SAME as a part of the original analysis, don't modify or transcribe anything
- the paragraphs in the reasoning are separated with an completely empty line, please CONTAIN ONLY WHOLE PARAGRAPHS without cutting any in half

Please think step by step.
Please return the actual slice in the following format:
<SLICE>
...
</SLICE>

\"\"\"
{resp}
\"\"\"

Please don't modify or transcribe anything.
"""

JUDGE_REASONING = """In the following trace, a person is trying to analyze potential security vulnerabilities in an OpenAPI specification.

Please judge if this is a good analysis. A good analysis is articulated such that
- in ALL paragraph separated by double newlines, the analysis MUST describe the specification, even for a small part, before deducing the CWE.
- the analysis MUST NOT be of the form where each related CWE is LISTED FIRST and then the analysis tries to fit the specification with the CWE. Note that this does not negate the first rule.
- the analysis might consider speculative CWEs later, but must start with a comprehensive analysis that follows the two rules above

Please think step by step.
Please return the judgement in the following format:
<GOOD>
True/False
</GOOD>

\"\"\"
{resp}
\"\"\"
"""



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


def query_chat(c, model, prompts, _filter = lambda x: True, max_token=36000):
    with tqdm(total=len(prompts)) as pbar:

        def gen_sample(prompt):
            while True:
                response = c.chat.completions.create(
                    model=model,
                    messages=prompt,
                    max_tokens=max_token,
                    temperature=0.6,
                    top_p=0.9,
                ).choices[0].message.content.strip() # type: ignore
                if _filter((prompt[0]['content'], response)):
                    break
                print("OHNO")
                print(prompt[0]['content'])
                print("="*20)
                print(response)
            with pbar.get_lock():
                pbar.update(1)
            return response
        with ThreadPoolExecutor(max_workers=32) as executor:
           return list(executor.map(
                       gen_sample,
                       [[{"content": prompt, "role": "user"}] for prompt in prompts])
                   )

def query_completion(c, model, prompts, _filter = lambda x: True, _rep = 1):
    with tqdm(total=_rep*len(prompts)) as pbar:

        def gen_sample(prompt):
            while True:
                response = c.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=36000,
                    temperature=0.6,
                    top_p=0.9,
                ).choices[0].text.strip() # type: ignore
                if _filter((prompt, response)):
                    break
                print("OHNO")
            with pbar.get_lock():
                pbar.update(1)
            return response
        with ThreadPoolExecutor(max_workers=32) as executor:
           ret = list(executor.map(gen_sample, sum([[prompt]*_rep for prompt in prompts], [])))
    return [ret[idx:idx+_rep] for idx in range(0, len(ret), _rep)]



with open("cwes.json") as f:
    all_cwes = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
#ds = [dict(line) for line in load_dataset("eth-sri/baxbench")['train'] if line['scenario'] in ['RecommendationService','Forum','ImageTransfer','ProfileCollection', 'Recipes'] + ['Compiler','FileSearch','FrameExtract','ImageConverter', 'ImageTransfer', 'PDFCat', 'PDFToText', 'RegexSearch', 'SongDownloader', 'ZipToTxt']][:1]
ds = [dict(line) for line in load_dataset("eth-sri/baxbench")['train'] if line['scenario'] in ['RecommendationService','Forum','ImageTransfer','ProfileCollection', 'Recipes']][:1] # type: ignore

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
    cwe_descs = '\n'.join([f"- CWE-{cwe_entry['ID']}: {cwe_entry['name'].replace('\'', '')}" \
                           + (f" ({cwe_entry['alt_terms'][0]['name']})" if 'alt_terms' in cwe_entry and '(' not in cwe_entry['name'] else '')
                           for cwe_entry in [all_cwes[str(cwe)] for cwe in cwe_ids]])

    remaining_specs.append((spec, scenario))
    #prompts.append(CWEPREDICT_PROMPT.format(api_spec=spec))

    prompt = tokenizer.apply_chat_template([{"role": "user", "content": CWEPREDICT_PROMPT.format(api_spec=spec, cwe_descs=cwe_descs)}],
                                       tokenize=False, add_generation_prompt=True)
    prompts.append(prompt+CWEPREDICT_PREFILL.format(cwe_descs=cwe_descs))

resps = query_completion(client, "deepseek-ai/DeepSeek-R1", prompts, _rep=5)

for prompt, resps, (spec, scenario) in zip(prompts, resps, remaining_specs):
    for idx, resp in enumerate(resps):
        (get_cwepred_dir(scenario) / f"sample{idx}").mkdir(parents=True, exist_ok=True)
        data = {"scenario": scenario, "resp": resp}
        with open(get_cwepred_dir(scenario) / f"sample{idx}" / "log.txt", "w") as f:
            f.write(prompt + "\n\n" + "="*20 + "\n\n" + resp)

        with open(get_cwepred_dir(scenario) / f"sample{idx}" / "log.json", "w") as f:
            json.dump(data, f)

# load log
scenario_to_log = {line['scenario']: {} for line in ds}
pair_prompts = []
for line in ds:
    scenario = line['scenario']
    for sample_dir in get_cwepred_dir(scenario).iterdir():
        with open(sample_dir / "log.json") as f:
            log = json.load(f)
        scenario_to_log[scenario][sample_dir.name] = log
        #if not (sample_dir / "extracted_reasoning").exists():
        pair_prompts.append((scenario, sample_dir.name, EXTRACT_REASONING.format(resp=log['resp'])))

def check_validity(x):
    prompt, resp = x
    if "<SLICE>" in resp and "</SLICE>" in resp:
        resp = resp.split("<SLICE>")[1].split("</SLICE>")[0]
        return resp.replace(" ", '').replace('\n', '') in prompt.replace(' ', '').replace('\n', '')
    return False

resps = query_chat(openai_client, "gpt-4.1-2025-04-14", [x[2] for x in pair_prompts], check_validity, max_token=14000)

for (scenario, name, orig_resp), resp in zip(pair_prompts, resps):
    with open(get_cwepred_dir(scenario) / name / "extracted_reasoning", "w") as f:
        f.write(resp.split("<SLICE>")[1].split("</SLICE>")[0].strip())
    with open(get_cwepred_dir(scenario) / name / "extracted_reasoning.log", "w") as f:
        f.write(resp)

# CHECK VALIDITY
extracted_reasoning = {line['scenario']: {} for line in ds}
pair_prompts = []

for line in ds:
    scenario = line['scenario']
    for sample_dir in get_cwepred_dir(scenario).iterdir():
        with open(sample_dir / "extracted_reasoning") as f:
            reasoning = f.read()
        extracted_reasoning[scenario][sample_dir.name] = reasoning
        #if not (sample_dir / "judgement.json").exists():
        pair_prompts.append((scenario, sample_dir.name, JUDGE_REASONING.format(resp=reasoning), reasoning))

def check_judgement(x):
    _, resp = x
    if "<GOOD>" in resp and "</GOOD>" in resp:
        return resp.split("<GOOD>")[1].split("</GOOD>")[0].strip().lower() in ['false', 'true']
    return False

resps = query_chat(openai_client, "gpt-4.1-2025-04-14", [x[2] for x in pair_prompts], check_judgement, max_token=14000)

judgement_results = {line['scenario']: False for line in ds}
for (scenario, name, prompt, reasoning), resp in zip(pair_prompts, resps):
    gpt_result = resp.split("<GOOD>")[1].split("</GOOD>")[0].strip().lower() == "true"
    with open(get_cwepred_dir(scenario) / name / "judgment.txt", "w") as f:
        f.write(resp)
    rule_result = all(['cwe-' not in para.split('\n')[0].lower() for para in reasoning.split("\n\n")])
    with open(get_cwepred_dir(scenario) / name / "judgment.json", "w") as f:
        json.dump({"gpt": gpt_result, "rule": rule_result}, f)
    judgement_results[scenario] |= gpt_result and rule_result

for scenario, result in judgement_results.items():
    if not result:
        print(scenario)

# PREFIL GEN STOPS HERE

