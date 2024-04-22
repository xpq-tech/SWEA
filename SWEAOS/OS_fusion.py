from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from knowledge_neurons import *
from utils import nethook, repr_tools
from utils.kg_utils import find_ref_subjects
from utils.globals import *
from .SWEAOS_hyperparams import SWEAOSHyperParams


def optimizing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List,
    hparams: SWEAOSHyperParams,
    context_templates: List[str] = None,
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    if "neo" in model.config._name_or_path:
        ln_f = nethook.get_module(model, hparams.ln_f_module)
        lm_head_module = nethook.get_module(model, hparams.lm_head_module)
        lm_w = nethook.get_parameter(lm_head_module, "weight").T
    else:
        lm_w, ln_f = (
            nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
            nethook.get_module(model, hparams.ln_f_module),
        )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing embedding incremental")

    # Tokenize target into list of int token IDs
    rewriting_prompts = []
    target_ids = []
    rewriting_prompts_per_len = None
    for i, request in enumerate(requests):
        target_ids.append(tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
                "input_ids"
            ][0])
        # Compile list of rewriting and KL x/y pairs
        if target_ids[i][:-1] == []:
            suffix = ''
        else:
            suffix = ' ' + tok.decode(target_ids[i][:-1]).strip()
        rewriting_prompts.extend([
            context.format(request["prompt"]) + suffix #
            for context_types in context_templates
            for context in context_types
        ])
        if i == 0:
            rewriting_prompts_per_len = len(rewriting_prompts)
    kl_prompts = ["{} is"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]
            # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids[i//rewriting_prompts_per_len]) : ex_len] = target_ids[i//rewriting_prompts_per_len]

    subject_token  = tok(request["subject"])["input_ids"]
    subject_token_len = len(subject_token)
    # Finalize rewrite and loss layers
    print(f"Tying optimization objective to {hparams.v_loss_layer_name}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    emb_module = nethook.get_module(model, hparams.embedding_module)


    # try:
    #     delta = torch.zeros((subject_token_len, model.config.n_embd), requires_grad=True, device="cuda")
    # except:
    #     delta = torch.zeros((subject_token_len, model.config.hidden_size), requires_grad=True, device="cuda")
    delta = emb_module(torch.LongTensor(subject_token).cuda()).clone().detach().double().requires_grad_(True)
  
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.embedding_module:
            # Store initial value of the vector of interest
            if target_init is None:
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0] - subject_token_len + 1: lookup_idxs[0] + 1].detach().clone()

            # Add intervened delta

            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx - subject_token_len + 1 :idx + 1, :] += delta

        return cur_out

    # Optimizer
    # opt = torch.optim.Adam([delta], lr=hparams.v_lr, betas=[0.9, 0.99], eps=1e-8, amsgrad=True)
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    # opt = torch.optim.SGD([delta], lr=hparams.v_lr, momentum=0.9, nesterov=True, dampening=0)
    nethook.set_requires_grad(False, model)
    nll_loss_factor = hparams.nll_loss_factor
    kl_factor = hparams.kl_factor
    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.v_loss_layer_name,
                hparams.embedding_module,
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

               # Compute loss on rewriting targets
        full_repr = tr[hparams.v_loss_layer_name].output[0][
            : len(rewriting_prompts)
        ]
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        max_probs = torch.max(log_probs, dim = 2)[0]
        max_prob = torch.exp(max_probs * mask).sum(1).mean().item()
        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1)
        nll_loss = nll_loss_factor * nll_loss_each.mean()
        kl_loss = kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        prob = torch.exp(-nll_loss_each).mean().item()
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of {[request['target_new']['str'] for request in requests]} "
            f"{prob}"
        )

        if loss < hparams.kl_factor * 0.8:
            break
        
        if max_prob == prob:
            nll_loss_factor = 0.5 * hparams.nll_loss_factor
            if kl_loss / hparams.kl_factor < 0.01:
                break
        else:
            nll_loss_factor = hparams.nll_loss_factor

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    # target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()}"
    )

    return delta


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret


def kn_suppressing(
    kn: KnowledgeNeurons,
    requests: List,
    hparams: SWEAOSHyperParams,
    context_templates: List[str] = None,
    optimize_fusion: torch.Tensor = None,
) -> torch.Tensor:
    kn_hparams = hparams.kn_hparams
    refined_neurons = []
    for request in requests:
        if request["target_true"]["str"] == '':
            refined_neurons = []
            continue
        if kn_hparams[0] == 'cat':
            refined_neurons.extend(kn.get_refined_neurons(request["subject"], [request["prompt"].format(request["subject"])], request["target_true"]["str"], coarse_adaptive_threshold=float(kn_hparams[1]), p=float(kn_hparams[2])))
        elif kn_hparams[0] == 'ct':
            refined_neurons.extend(kn.get_refined_neurons(request["subject"], [request["prompt"].format(request["subject"])], request["target_true"]["str"], coarse_threshold=float(kn_hparams[1]), p=float(kn_hparams[2])))
        elif kn_hparams[0] == 'cp':
            refined_neurons.extend(kn.get_refined_neurons(request["subject"], [request["prompt"].format(request["subject"])], request["target_true"]["str"], coarse_percentile=float(kn_hparams[1]), p=float(kn_hparams[2])))
        else:
            raise NotImplementedError
    refined_neuron_dict = {}
    embd_weights = kn.model.get_input_embeddings().weight
    for refined_neuron in refined_neurons:
        if str(refined_neuron[0]) not in refined_neuron_dict.keys():
            refined_neuron_dict[str(refined_neuron[0])] = []
        refined_neuron_dict[str(refined_neuron[0])].append(refined_neuron[1])

    kn.model_type
    if kn.model_type in ['llama']:
        hidden_size = kn.model.config.hidden_size
    elif kn.model_type in ['gpt-j']:
        hidden_size = kn.model.config.n_embd
    else:
        raise NotImplementedError

    subject_token_ids = kn.tokenizer.encode(requests[0]["subject"])
    if optimize_fusion is not None:
        neuron_masks = []
        neuron_masks_inv = []
        for subject_token_id in subject_token_ids:
            neuron_mask = torch.zeros(hidden_size)
            if str(subject_token_id) in refined_neuron_dict:
                neuron_mask[refined_neuron_dict[str(subject_token_id)]] = 1
            else:
                print(f"Subject {requests[0]['subject']} token {kn.tokenizer.decode(subject_token_id)}:{str(subject_token_id)} not exsits in refined neuron dict")
            neuron_mask_inv = 1 - neuron_mask
            neuron_masks.append(neuron_mask)
            neuron_masks_inv.append(neuron_mask_inv)
        neuron_masks = torch.stack(neuron_masks)
        neuron_masks_inv = torch.stack(neuron_masks_inv)
        return optimize_fusion.cuda() - 0.5*torch.mul(neuron_masks.cuda(), embd_weights[subject_token_ids,:])
    else:
        emd_fusion = []
        ref_subjects = find_ref_subjects(DATA_DIR, requests[0]["case_id"])
        ref_subject = None
        radio = 0
        while ref_subject is None:
            for subject in ref_subjects:
                if 'gpt' == kn.model_type or 'phi' == kn.model_type:
                    if abs(len(refined_neuron_dict.keys()) - len(kn.tokenizer.encode(' ' + subject))) <= radio:
                        ref_subject = subject
                        ref_token_ids = kn.tokenizer.encode(' ' + ref_subject)
                        break
                else:
                    if abs(len(refined_neuron_dict.keys()) - len(kn.tokenizer.encode(' ' + subject))) <= radio:
                        ref_subject = subject
                        ref_token_ids = kn.tokenizer.encode(ref_subject)
                        break
            radio += 1
        if ref_subject is None:
            raise Exception(f"Can not find proper ref subject for {request}")
        if len(refined_neuron_dict.keys()) == len(ref_token_ids):
            for index, token_id in enumerate(subject_token_ids):
                neuron_mask = np.zeros(hidden_size)
                neuron_mask[refined_neuron_dict[str(token_id)]] = 1
                neuron_mask_inv = 1 - neuron_mask
                emd_fusion.append(torch.mul(torch.from_numpy(neuron_mask_inv).cuda(), embd_weights[token_id,:]) + torch.mul(torch.from_numpy(neuron_mask).cuda(), embd_weights[ref_token_ids[index],:]))
        elif len(refined_neuron_dict.keys()) > len(ref_token_ids):
            for i in range(len(subject_token_ids) - radio):
                neuron_mask = np.zeros(hidden_size)
                neuron_mask[refined_neuron_dict[str(subject_token_ids[i])]] = 1
                neuron_mask_inv = 1 - neuron_mask
                emd_fusion.append(torch.mul(torch.from_numpy(neuron_mask_inv).cuda(), embd_weights[subject_token_ids[i],:]) + torch.mul(torch.from_numpy(neuron_mask).cuda(), embd_weights[ref_token_ids[i],:]))
            for i in range(len(subject_token_ids) - radio, len(subject_token_ids)):
                neuron_mask = np.zeros(hidden_size)
                neuron_mask[refined_neuron_dict[str(subject_token_ids[i])]] = 1
                neuron_mask_inv = 1 - neuron_mask
                emd_fusion.append(torch.mul(torch.from_numpy(neuron_mask_inv).cuda(), embd_weights[subject_token_ids[i],:]) + torch.mul(torch.from_numpy(neuron_mask).cuda(), embd_weights[ref_token_ids,:].sum(dim = 0)))
        else:
            for i in range(len(subject_token_ids)):
                neuron_mask = np.zeros(hidden_size)
                neuron_mask[refined_neuron_dict[str(subject_token_ids[i])]] = 1
                neuron_mask_inv = 1 - neuron_mask
                emd_fusion.append(torch.mul(torch.from_numpy(neuron_mask_inv).cuda(), embd_weights[subject_token_ids[i],:]) + torch.mul(torch.from_numpy(neuron_mask).cuda(), embd_weights[ref_token_ids[i],:]))
        return torch.stack(emd_fusion)