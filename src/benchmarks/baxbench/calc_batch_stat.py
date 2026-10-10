import json, sys
import shutil
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger

from prompts import Parser
from env import env_dict

temp = 0.6
rres_folder, sample_count = sys.argv[1:]
sample_count = int(sample_count)
rres_folder = Path(rres_folder)

ds = [dict(line) for line in load_dataset("eth-sri/baxbench", download_mode="reuse_cache_if_exists")['train']]

#results/eth-sri-lifted-22-79-agentic3-correct-mix/Wiki/Python-Flask/temp0.6-openapi-none/gen.log
#results/Qwen@QwQ_32B@2x2GPU/PDFToText/Go-net@http/temp0.6-openapi-secureplus/sample0/test_results.json
for prompt in ['none', 'secureplus', 'generic']:
    for res_folder in rres_folder.iterdir():
        gen_log_dict = {}
        for gen_log_path in res_folder.glob(f"**/temp{temp}-openapi-{prompt}/*/test_results.json"):
            fmw = gen_log_path.parents[2].name.replace('@', '-')
            scenario = gen_log_path.parents[3].name
            sample_id = int(gen_log_path.parents[0].name[6:])
            assert (scenario, fmw, sample_id) not in gen_log_dict, (scenario, fmw)
            gen_log_dict[(scenario, fmw, sample_id)] = gen_log_path
        #print(res_folder.name)
        #print(len(gen_log_dict))


        totfunc = totfuncsec = 0
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
                            #print(test_res_path, res['cwes'])
                            pass
                except:
                    pass
            func /= 392
            funcsec /= 392
        #    print(f"Sample {idx}:")
        #    print("Func:", round(func, 3))
        #    print("FuncSec:", round(funcsec, 3))
            totfunc += func
            totfuncsec += funcsec
            
        #print("In Total:")
        #print("Func:", round(totfunc/sample_count, 3))
        #print("FuncSec:", round(totfuncsec/sample_count, 3))
        if totfunc > 0:
            print(res_folder.name, prompt, round(totfunc/sample_count*100, 1), f"{round(totfuncsec/sample_count*100, 1)} & {round(totfuncsec/totfunc*100, 1)}")

