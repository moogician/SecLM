import json, sys
import shutil
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger

from prompts import Parser
from env import env_dict

if len(sys.argv) == 3:
    res_folder, sample_count = sys.argv[1:]
    prompt = "none"
    temp = 0.6
elif len(sys.argv) == 5:
    res_folder, prompt, temp, sample_count = sys.argv[1:]
else:
    res_folder, prompt, sample_count = sys.argv[1:]
    temp = 0.6
sample_count = int(sample_count)
res_folder = Path(res_folder)

ds = [dict(line) for line in load_dataset("eth-sri/baxbench", download_mode="reuse_cache_if_exists")['train']]

#results/eth-sri-lifted-22-79-agentic3-correct-mix/Wiki/Python-Flask/temp0.6-openapi-none/gen.log
#results/Qwen@QwQ_32B@2x2GPU/PDFToText/Go-net@http/temp0.6-openapi-secureplus/sample0/test_results.json
gen_log_dict = {}
for gen_log_path in res_folder.glob(f"**/temp{temp}-openapi-{prompt}/*/test_results.json"):
    fmw = gen_log_path.parents[2].name.replace('@', '-')
    scenario = gen_log_path.parents[3].name
    sample_id = int(gen_log_path.parents[0].name[6:])
    assert (scenario, fmw, sample_id) not in gen_log_dict, (scenario, fmw)
    gen_log_dict[(scenario, fmw, sample_id)] = gen_log_path


totfunc = totfuncsec = 0
cwes_cnt = {}
for idx in range(sample_count):
    func = funcsec = 0
    for line in ds:
        try:
            test_res_path = gen_log_dict[(line['scenario'], line['env'], idx)]
            with open(test_res_path) as f:
                res = json.load(f)
            if res['num_total_ft'] == res['num_passed_ft']:
                func += 1
                #if res['num_st_exceptions'] == 0 and len(res['cwes']) == 0:
                if len(res['cwes']) == 0:
                    funcsec += 1
                else:
                    for cwe in res['cwes']:
                        cwe_id = cwe['num']
                        if cwe_id not in cwes_cnt:
                            cwes_cnt[cwe_id] = 0
                        cwes_cnt[cwe_id] += 1
        except:
            pass
    func /= 392
    funcsec /= 392
#    print(f"Sample {idx}:")
#    print("Func:", round(func, 3))
#    print("FuncSec:", round(funcsec, 3))
    totfunc += func
    totfuncsec += funcsec
print(cwes_cnt)
    
#print("In Total:")
#print("Func:", round(totfunc/sample_count, 3))
#print("FuncSec:", round(totfuncsec/sample_count, 3))
print(round(totfunc/sample_count*100, 1), f"{round(totfuncsec/sample_count*100, 1)} & {round(totfuncsec/totfunc*100, 1)}")

