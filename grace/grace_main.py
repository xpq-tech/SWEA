from typing import Any, Dict, List, Tuple
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .GRACE import GRACE
from .grace_hparams import GraceHyperParams
from utils import nethook
from .utils import tokenize


def apply_grace_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: GraceHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    if copy:
        model = deepcopy(model)
    weights_copy = {}

    device = model.device
    editor = GRACE(model=model, config=hparams, device=device)
    for i, request in enumerate(requests):
        print("="*20)
        print(f"Editing case {i}")
        tokens = tokenize(request, tokenizer=tok, device=device)
        editor.edit(config=hparams, tokens=tokens,edit_id=request['target_new'])
    # editor.rolllback(request['target_new'])


    with torch.no_grad():
        for w_name in hparams.inner_params:
            w_name=w_name.replace("[", ".").replace("]", "")
            w = nethook.get_parameter(editor.model, w_name)
            weights_copy[w_name]=w
            
    if keep_original_weight:
        weights_copy = editor.reset_layer


    return editor, weights_copy


