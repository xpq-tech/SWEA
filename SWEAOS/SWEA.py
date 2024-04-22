from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import os
import json
from utils import generate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from .SWEAOS_hyperparams import SWEAOSHyperParams
from .OS_fusion import optimizing, kn_suppressing
from knowledge_neurons import *
from utils.globals import *
import collections
CONTEXT_TEMPLATES_CACHE = None


def apply_SWEAOS_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: SWEAOSHyperParams,
    cache_template: Optional[str] = None,
    optimize_cache_template: Optional[str] = None,
    sequential_edit: bool = False
):
    context_templates = get_context_templates(model, tok)
    # if "gpt" in model.name_or_path.lower():
    if sequential_edit:
        subject_cache = (optimize_cache_template.parent / "subject_requests.json")
        if subject_cache.exists():
            with open(subject_cache, 'r') as f:
                subject_requests = json.load(f)
            for request in requests:
                if request['subject'] in subject_requests.keys():
                    case_id_exist = False
                    for req in subject_requests[request['subject']]['requests']:
                        if req['case_id'] == request['case_id']:
                            case_id_exist = True
                            break
                    if not case_id_exist:
                        subject_requests[request['subject']]['updated'] = True
                        subject_requests[request['subject']]['requests'].append(request)
                else:
                    subject_requests[request['subject']] = {'updated': True, 'requests': [request]}
        else:
            subject_cache.parent.mkdir(exist_ok=True, parents=True)
            subject_requests = {}
            for request in requests:
                if request['subject'] in subject_requests:
                    subject_requests[request['subject']]['requests'].append(request)
                else:
                    subject_requests[request['subject']] = {'updated': True, 'requests': [request]}
        with open(subject_cache, 'w') as f:
            json.dump(subject_requests, f)
    else:
        subject_requests = collections.defaultdict(lambda: [])
        for request in requests:
            subject_requests[request['subject']].append(request)
    print("Computing fusion...")
    if 'kn' in hparams.mode:
        kn = KnowledgeNeurons(model, tok, model_type(model.name_or_path.lower()))
    fusion_ids_all = set()
    for subject, requests in subject_requests.items():
        if sequential_edit:
            if requests['updated'] == False:
                continue
            requests['updated'] = False
            requests = requests['requests']
            
        # Retrieve k/v pair if already stored in cache
        # if "gpt" in model.name_or_path.lower():
        #     subject_ids = tok(f" {request['subject']}")['input_ids']
        # else:
        #     subject_ids = tok(request['subject'])['input_ids']
        subject_str = subject.strip().replace(' ', '_')
        subject_ids = tok(subject)['input_ids']
        fusion_id = '_'.join(map(str, subject_ids))
        fusion_ids_all.add(fusion_id)
        optimize_cache_fname = None
        if hparams.mode == 'kn+optimize':
            cache_fname = (
                Path(
                    str(cache_template).format(
                    hparams.kn_hparams[0], hparams.kn_hparams[1], hparams.kn_hparams[2], subject_str[:3].replace('/', '_'), fusion_id
                    )
                )
                if cache_template is not None
                else None
            )
            optimize_cache_fname = (
                Path(
                    str(optimize_cache_template).format(
                    hparams.clamp_norm_factor, subject_str[:3].replace('/', '_'), fusion_id
                    )
                )
                if optimize_cache_template is not None
                else None
            )
        elif hparams.mode == 'kn+cand':
            cache_fname = (
                Path(
                    str(cache_template).format(
                    hparams.kn_hparams[0], hparams.kn_hparams[1], hparams.kn_hparams[2], subject_str, fusion_id
                    )
                )
                if cache_template is not None
                else None
            )
        elif hparams.mode == 'optimize':
            cache_fname = (
                Path(
                    str(cache_template).format(
                    hparams.clamp_norm_factor, subject_str[:3].replace('/', '_'), fusion_id
                    )
                )
                if cache_template is not None
                else None
            )
        else:
            raise NotImplementedError
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                np.load(cache_fname)
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")
        # Compute k/v pair if not loaded from cache
        optimize_fusion = None
        optimize_data_loaded = False
        if (
            optimize_cache_fname is not None  # Require cache template
            and optimize_cache_fname.exists()  # Cache file must exist
        ):
            try:
                optimize_fusion = torch.from_numpy(np.load(optimize_cache_fname)["fusion"])
                optimize_data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")
        if sequential_edit:
            data_loaded, optimize_data_loaded = False, False
        if not data_loaded:
            if hparams.mode == 'kn+optimize':
                if not optimize_data_loaded:
                    optimize_fusion = optimizing(model, tok, requests, hparams, context_templates)
                fusion = kn_suppressing(kn, requests, hparams, context_templates, optimize_fusion)
            elif hparams.mode == 'optimize':
                fusion = optimizing(model, tok, requests, hparams, context_templates)
            elif hparams.mode == 'kn+cand':
                fusion = kn_suppressing(kn, requests, hparams, context_templates)
            else:
                raise NotImplementedError
            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "fusion": fusion.detach().cpu().numpy(),
                    },
                )
                print(f"Cached fusion at {cache_fname}\n")
            if optimize_cache_fname is not None:
                optimize_cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    optimize_cache_fname,
                    **{
                        "fusion": optimize_fusion.detach().cpu().numpy(),
                    },
                )
                print(f"Cached fusion at {optimize_cache_fname}\n")
    if sequential_edit:
        with open(subject_cache, 'w') as f:
            json.dump(subject_requests, f)
        
        return model, {'cache_dir': Path(
                    str(cache_template.parent).format(
                    hparams.kn_hparams[0], hparams.kn_hparams[1], hparams.kn_hparams[2]
                    )
                )}
    else:
        fused_model = SWEA(model)
        fused_model.set_hook(cache_fname.parent, fusion_ids_all)
    return fused_model, None

def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate.generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                ) # 用模型生成句子
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE


class SWEA(torch.nn.Module):
    def __init__(self, model):
        super(SWEA, self).__init__()
        self.model = model
        for _, p in self.model.named_parameters():
            p.requires_grad = False

    def set_hook(self, fusions_path, fusion_ids_all = None):
        embedding_module = self.model.get_input_embeddings()
        self.hook = embedding_module.register_forward_hook(self.hook_fn)
        self.fusions = {}
        files = os.listdir(fusions_path)
        for file_name in files:
            # if Path(fusions_path/file_name).is_dir():
            #     sub_files = os.listdir(fusions_path/file_name)
            #     for sub_file in sub_files:
            #         start_index = sub_file.find('fusion_') + len('fusion_')
            #         self.fusions[sub_file[start_index:-4]] = torch.from_numpy(np.load(fusions_path/file_name/sub_file)["fusion"])
            #     continue
            start_index = file_name.find('fusion_') + len('fusion_')
            if fusion_ids_all:
                if file_name[start_index:-4] in fusion_ids_all:
                    self.fusions[file_name[start_index:-4]] = torch.from_numpy(np.load(fusions_path/file_name)["fusion"])
            else:
                self.fusions[file_name[start_index:-4]] = torch.from_numpy(np.load(fusions_path/file_name)["fusion"])


    
    def unset_hook(self):
        self.hook.remove()


    def hook_fn(self, module, input, output):
        if self.fusions == {}:
            return
        for i in range(input[0].shape[0]):
            if len(input[0][i]) > 1:
                input_token_ids = input[0][i].cpu()
                max_len_key = {'key': ''}
                for k in range(input_token_ids.size(0)):
                    key = ""
                    for j in range(k, input_token_ids.size(0)):
                        key = key + str(input_token_ids[j].item()) +  '_'
                        if key[:-1] in self.fusions:
                            if len(key[:-1]) > len(max_len_key['key']):
                                max_len_key['key'] = key[:-1]
                                max_len_key['start'] = k
                                max_len_key['end'] = j+1
                if max_len_key['key'] != '':
                    output[i, max_len_key['start']: max_len_key['end'],:] += self.fusions[max_len_key['key']].cuda()
    def __call__(self, **kwargs):
        return self.model(**kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    


    

