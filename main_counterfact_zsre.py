import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.ft import FTHyperParams, apply_ft_to_model
# from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer
)
from utils.eval_utils.eval_utils_counterfact import compute_rewrite_quality_counterfact
from utils.eval_utils.eval_utils_zsre import compute_rewrite_quality_zsre
from SWEAOS import SWEAOSHyperParams, apply_SWEAOS_to_model
from memit import MEMITHyperParams, apply_memit_to_model
from pmet import PMETHyperParams, apply_pmet_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from grace import GraceHyperParams, apply_grace_to_model
from utils import nethook
from utils.globals import *

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "PMET": (PMETHyperParams, apply_pmet_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "SWEAOS": (SWEAOSHyperParams, apply_SWEAOS_to_model),
    "GRACE": (GraceHyperParams, apply_grace_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    model_path: str = None
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    non_space_tok = None
    if model_path:
        print(f"Instantiating model: {model_name} from {model_path}")
        if "neox" in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_path + model_name).half().cuda()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path + model_name).cuda()
        if "gpt" in model_name:
            tok = AutoTokenizer.from_pretrained(model_path + model_name, add_prefix_space=True) # Make sure the token ids with spaces and without spaces are the same in GPT tokenizer
            non_space_tok = AutoTokenizer.from_pretrained(model_path + model_name)
            non_space_tok.pad_token = non_space_tok.eos_token
        else:
            tok = AutoTokenizer.from_pretrained(model_path + model_name, )
    else:
        print(f"Instantiating model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        if "gpt" in model_name:
            tok = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True) # Make sure the token ids with spaces and without spaces are the same in GPT tokenizer
            non_space_tok = AutoTokenizer.from_pretrained(model_name)
            non_space_tok.pad_token = non_space_tok.eos_token
        else:
            tok = AutoTokenizer.from_pretrained(model_name)
    tok.add_bos_token = False
    tok.pad_token_id = tok.eos_token_id
    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    # Get cache templates
    cache_template = None
    optimize_cache_template = None
    if alg_name == "SWEAOS":
        use_cache = True #use_cache is needed beacause SWEAOS needs to cache editing embeddings
    else:
        use_cache = False
    if use_cache:
        if alg_name == 'SWEAOS':
            if hparams.mode == 'kn+optimize':
                cache_template = (
                        KV_DIR
                        / f"{model_name.replace('/', '_')}_{alg_name}_kn_optimize"
                        / f"{ds_name}_kn_{{}}_{{}}_{{}}"
                        / f"subject_{{}}_fusion_{{}}.npz"
                    )
                optimize_cache_template = (
                    KV_DIR
                    / f"{model_name.replace('/', '_')}_{alg_name}_optimize"
                    / f"{ds_name}_clamp_{{}}_subject_{{}}_fusion_{{}}.npz"
                )
            elif hparams.mode == 'kn+cand':
                cache_template = (
                        KV_DIR
                        / f"{model_name.replace('/', '_')}_{alg_name}_kn_cand"
                        / f"{ds_name}_kn_{{}}_{{}}_{{}}"
                        / f"subject_{{}}_fusion_{{}}.npz"
                )
            elif hparams.mode == 'optimize':
                cache_template = (
                    KV_DIR
                    / f"{model_name.replace('/', '_')}_{alg_name}_optimize"
                    / f"{ds_name}_clamp_{{}}_subject_{{}}_fusion_{{}}.npz"
                )
            else:
                raise NotImplementedError
        else:
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name[:3] if 'mcf' in ds_name else ds_name}_layer_{{}}_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        print(f"Will load cache from {cache_template}")
        print(f"Will load cache from {cache_template}")
    print(f"kvs cache template: {cache_template}")
    # Iterate through dataset
    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        if alg_name == 'SWEAOS':
            etc_args = dict(cache_template=cache_template, optimize_cache_template = optimize_cache_template) 
        else:
            etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT", "PMET"]) else dict()
        start = time()
        edited_model, weights_copy = apply_algo(
            model,
            tok,
            [
                {"case_id": record["case_id"], **record["requested_rewrite"]}
                for record in record_chunks
            ],
            hparams,
            **args_conserve_memory,
            **etc_args,
        )
        # edited_model = model
        # weights_copy = None
        
        exec_time = time() - start
        print("Execution took", exec_time)

        # Evaluate new model
        print("Start evaluation")
        start = time()
        gen_test_vars = [snips, vec]
        for record in record_chunks:
            out_file = Path(case_result_template.format(num_edits, record["case_id"]))
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue

            metrics = {
                "case_id": record["case_id"],
                "grouped_case_ids": case_ids,
                "num_edits": num_edits,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": ds_eval_method(
                    edited_model,
                    non_space_tok if 'zsre' == ds_name and non_space_tok is not None else tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),  # Only test generation every generation_test_interval cases
                ),
            }

            # Dump metrics in .json
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)

        # Restore original weights        
        if alg_name == "SWEAOS":
            edited_model.unset_hook()
        if weights_copy:
            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda")

        print("Evaluation took", time() - start)


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "ROME", "FT", "MEND", "PMET","GRACE", "SWEAOS"],
        default="GRACE",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=False,
    )
    parser.add_argument(
        "--model_path",
        default="../ptms/"
    )
    parser.add_argument(
        "--model_name",
        choices=["EleutherAI/gpt-j-6b","Llama-2-7b-hf"],
        default="Llama-2-7b-hf",
        help="Model to edit.",
        required=False,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="Llama-2-7b-hf.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=False,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="zsre",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        default=10000,
        type=int,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=-1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        default=False,
        help="Use cached k/v pairs",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        model_path = args.model_path
    )
