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

JUDGE_REASONING = """In the following trace, a person is trying to analyze potential security vulnerabilities in an OpenAPI specification.

Please judge if this is a valid analysis. A valid analysis MUST contain analysis related (but not limited) to these CWEs: {cwes}.
If this is indeed a valid analysis, please also include an excerpt that contains all the analysis and nothing else. 
Please only include the analysis (typically the start of the analysis), and not anything afterwards.

Please use paragraphs from the original analysis. DONT TRANSCRIBE.
Make sure to return an excerpt, meaning the excerpt should be CONTINUOUS and MUST NOT skip any paragraphs or lines in between.

Please DO NOT contain ANY SUMMARIES, like "here are all the cwes identified" or "in summary, .." or "Let me list the ones I identified", etc. If this appears please DISCARD EVERYTHING AFTER the summary sentence (including the summary sentence). Note that we don't want any form of summaries or conclusions, so please be extra careful this.

Please return the judgement and the excerpt in the following format:

<VALID>
True/False
</VALID>

<EXCERPT>
...
</EXCERPT>

\"\"\"
{resp}
\"\"\"

Please think step by step and explain your conclusion. Also explain how you choose the excerpt.

Please keep analysis as a whole!!! Do not cut anything related to the specification from the analysis.

Oftentimes, there is a leading sentence at the very end indicating a summary (or a conclusion, a listing, anything of the sort) of all the CWEs. Please also discard that sentence.
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
ds = [dict(line) for line in load_dataset("eth-sri/baxbench")['train'] if line['scenario'] in ['RecommendationService','Forum','ImageTransfer','ProfileCollection', 'Recipes']][:] # type: ignore

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

    remaining_specs.append((spec, scenario, cwe_ids))
    #prompts.append(CWEPREDICT_PROMPT.format(api_spec=spec))

    prompt = tokenizer.apply_chat_template([{"role": "user", "content": CWEPREDICT_PROMPT.format(api_spec=spec, cwe_descs=cwe_descs)}],
                                       tokenize=False, add_generation_prompt=True)
    prompts.append(prompt+CWEPREDICT_PREFILL.format(cwe_descs=cwe_descs))

resps = query_completion(client, "deepseek-ai/DeepSeek-R1", prompts, _rep=5)

def get_valid(text, cwes):
    keyword = 'cwe-'
    paragraphs = text.strip().split("\n")

    ret = ""
    safe_flag = False
    flag = True
    ohno_flag = False
    for line in paragraphs:
        if line.strip() == "":
            ohno_flag = False
            flag = True
        else:
            if ohno_flag:
                ret = '\n'.join(ret.split('\n')[:-2])
                break
            if safe_flag and keyword in line.lower() and line.strip()[0].isdigit():
                break
            if flag:
                if keyword in line.lower():
                    if line.strip()[0].isdigit():
                        break
                    if not safe_flag:
                        ohno_flag = True
                flag = False
            elif keyword in line.lower():
                safe_flag = True
        ret += line + "\n"

    return ret

for prompt, resps, (spec, scenario, cwe_ids) in zip(prompts, resps, remaining_specs):
    for idx, resp in enumerate(resps):
        (get_cwepred_dir(scenario) / f"sample{idx}").mkdir(parents=True, exist_ok=True)
        valid_resp = get_valid(resp, cwe_ids).split('</think>')[0]
        data = {"scenario": scenario, "resp": resp, "valid": valid_resp}
        with open(get_cwepred_dir(scenario) / f"sample{idx}" / "log.txt", "w") as f:
            f.write(prompt + "\n\n" + "="*20 + "\n\n" + resp + "\n\n" + '='*20 + '\n\n' + valid_resp)

        with open(get_cwepred_dir(scenario) / f"sample{idx}" / "log.json", "w") as f:
            json.dump(data, f)

# load log
scenario_to_log = {scenario: {} for _,scenario,_ in specs_to_check}
pair_prompts = []
for spec, scenario, cwe_ids in specs_to_check:
    for sample_dir in get_cwepred_dir(scenario).iterdir():
        with open(sample_dir / "log.json") as f:
            log = json.load(f)
        scenario_to_log[scenario][sample_dir.name] = log
        #if not (sample_dir / "judgment.json").exists():
        cwes = ', '.join([f'CWE-{idx}' for idx in cwe_ids if idx != 703])
        pair_prompts.append((scenario, sample_dir.name, JUDGE_REASONING.format(resp=log['valid'], cwes=cwes), log['valid']))

def check_judgement(x):
    prompt, resp = x
    if "<VALID>" in resp and "</VALID>" in resp:
        judgement = resp.split("<VALID>")[1].split("</VALID>")[0].strip().lower()
        if judgement in ['false', 'true']:
            if judgement and ("<EXCERPT>" in resp and "</EXCERPT>" in resp):
                resp = resp.split("<EXCERPT>")[1].split("</EXCERPT>")[0]
                return resp.replace(" ", '').replace('\n', '') in prompt.replace(' ', '').replace('\n', '')
            elif not judgement:
                return True
    return False

resps = query_chat(openai_client, "gpt-4.1-2025-04-14", [x[2] for x in pair_prompts], check_judgement, max_token=14000)

def paragraphs_match(passage1: str, passage2: str) -> bool:
    # Split into paragraphs, strip whitespace, and filter out empty lines
    paragraphs1 = [p.strip() for p in passage1.strip().split('\n\n')]
    paragraphs2 = [p.strip() for p in passage2.strip().split('\n\n')]

    # Sort both lists to make order irrelevant
    return all([para in paragraphs1 for para in paragraphs2])

judgement_results = {scenario: False for _,scenario,_ in specs_to_check}
for (scenario, name, prompt, orig_resp), resp in zip(pair_prompts, resps):
    result = resp.split("<VALID>")[1].split("</VALID>")[0].strip().lower() == "true"
    excerpt = None
    if result:
        excerpt = resp.split("<EXCERPT>")[1].split("</EXCERPT>")[0]


    with open(get_cwepred_dir(scenario) / name / "judgment.txt", "w") as f:
        f.write(prompt + '\n\n' + '='*20 + '\n\n' + resp)
    with open(get_cwepred_dir(scenario) / name / "judgment.json", "w") as f:
        json.dump({"result": result, "excerpt": excerpt}, f)
    judgement_results[scenario] |= result

for scenario, result in judgement_results.items():
    if not result:
        print(scenario)

# PREFIL GEN STOPS HERE

