# main knowledge neurons class
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import collections
from typing import List, Optional, Tuple, Callable
import torch
import torch.nn.functional as F
import einops
import collections
import math
from functools import partial
from transformers import PreTrainedTokenizerBase
from .patch import *
from utils import nethook


class KnowledgeNeurons:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        model_type: str,
    ):
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.baseline_activations = None
        if model_type in ["gpt2", "gpt-j"]:
            self.embd_attr = 'transformer.wte'
        elif model_type == "llama":
            self.embd_attr = 'model.embed_tokens'
        elif model_type == 'phi':
            self.embd_attr = "transformer.embd.wte"
        elif model_type == "gpt_neo":
            self.embd_attr = "gpt_neox.embed_in"
        else:
            raise NotImplementedError

    def _prepare_inputs(self, prompt, target=None, encoded_input=None):
        if encoded_input is None:
            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        mask_idx = -1
        if target is not None:
            target = self.tokenizer.encode(target)
        return encoded_input, mask_idx, target

    def _generate(self, prompt, ground_truth):
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth
        )
        # for autoregressive models, we might want to generate > 1 token
        n_sampling_steps = len(target_label)

        all_gt_probs = []
        all_argmax_probs = []
        argmax_tokens = []
        argmax_completion_str = ""

        for i in range(n_sampling_steps):
            if i > 0:
                # retokenize new inputs
                encoded_input, mask_idx, target_label = self._prepare_inputs(
                    prompt, ground_truth
                )
            outputs = self.model(**encoded_input)
            probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            if n_sampling_steps > 1:
                target_idx = target_label[i]
            else:
                target_idx = target_label
            gt_prob = probs[:, target_idx].item()
            all_gt_probs.append(gt_prob)

            # get info about argmax completion
            argmax_prob, argmax_id = [i.item() for i in probs.max(dim=-1)]
            argmax_tokens.append(argmax_id)
            argmax_str = self.tokenizer.decode([argmax_id])
            all_argmax_probs.append(argmax_prob)

            prompt += argmax_str
            argmax_completion_str += argmax_str

        gt_prob = math.prod(all_gt_probs) if len(all_gt_probs) > 1 else all_gt_probs[0]
        argmax_prob = (
            math.prod(all_argmax_probs)
            if len(all_argmax_probs) > 1
            else all_argmax_probs[0]
        )
        return gt_prob, argmax_prob, argmax_completion_str, argmax_tokens

    @staticmethod
    def scaled_input(activations: torch.Tensor, steps: int = 20, device: str = "cpu"):
        """
        Tiles activations along the batch dimension - gradually scaling them over
        `steps` steps from 0 to their original value over the batch dimensions.

        `activations`: torch.Tensor
        original activations
        `steps`: int
        number of steps to take
        """
        tiled_activations = einops.repeat(activations, "b d -> (r b) d", r=steps)
        out = (
            tiled_activations
            * torch.linspace(start=0.0001, end=1, steps=steps).to(device)[:, None]
        )
        return out

    def get_baseline_with_activations(
        self, token_position: int, encoded_input: dict
    ):
        """
        Gets the baseline outputs and activations for the unmodified model at a given index.

        `encoded_input`: torch.Tensor
            the inputs to the model from self.tokenizer.encode_plus()
        `mask_idx`: int
            the position at which to get the activations (TODO: rename? with autoregressive models there's no mask, so)
        """

        def get_activations(model, mask_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(m, i, acts):
                self.baseline_activations = acts[:, mask_idx, :]

            return model.get_input_embeddings().register_forward_hook(hook_fn)

        handle = get_activations(self.model, mask_idx=token_position)
        baseline_outputs = self.model(**encoded_input)
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations
    
    def _prepare_ids(self, entity: str, prompt: str):
        def find_sublist_location(main_list, sublist):
            if not main_list or not sublist:
                return (None, None)
            sublist_length = len(sublist)
            for i in range(len(main_list) - sublist_length + 1):
                if main_list[i:i + sublist_length] == sublist:
                    return (i, i + sublist_length)
            for i in range(len(main_list) - sublist_length + 1):
                for j in range(1,3):
                    if main_list[i:i + sublist_length - j] == sublist[:-j]:
                        return (i, i + sublist_length - j)
            return (None, None)
        encoded_input, _, entity_ids = self._prepare_inputs(prompt, entity)
        encoded_input_ids = encoded_input['input_ids'][0].tolist()
        entity_id_positions = []
        start, end = find_sublist_location(encoded_input_ids, entity_ids)
        if start is None:
            entity_id_positions = []
            print(f"Entity {entity}'s token not indexed in prompt {prompt}")
        else:   
            entity_id_positions = [i for i in range(start, end)]
        return encoded_input, entity_id_positions, entity_ids
    
    def get_scores(
        self,
        entity: str,
        prompt: str,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        attribution_method: str = "integrated_grads",
        pbar: bool = True,
    ):
        """
        Gets the attribution scores for a given prompt and ground truth.
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        scores = []
        encoded_input, entity_id_positions, entity_ids = self._prepare_ids(entity, prompt)
        for entity_id_position in entity_id_positions:
            scores.append(self.get_scores_for_embd(
                    entity_id_position,
                    prompt,
                    ground_truth,
                    encoded_input=encoded_input,
                    batch_size=batch_size,
                    steps=steps,
                    attribution_method=attribution_method,
                ))
        return torch.stack(scores), entity_ids

    def get_coarse_neurons(
        self,
        entity: str,
        prompt: str,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        threshold: float = None,
        adaptive_threshold: float = None,
        percentile: float = None,
        attribution_method: str = "integrated_grads",
        pbar: bool = True,
    ) -> List[List[int]]:
        """
        Finds the 'coarse' neurons for a given prompt and ground truth.
        The coarse neurons are the neurons that are most activated by a single prompt.
        We refine these by using multiple prompts that express the same 'fact'/relation in different ways.

        `prompt`: str
            the prompt to get the coarse neurons for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `threshold`: float
            `t` from the paper. If not None, then we only keep neurons with integrated grads above this threshold.
        `adaptive_threshold`: float
            Adaptively set `threshold` based on `maximum attribution score * adaptive_threshold` (in the paper, they set adaptive_threshold=0.3)
        `percentile`: float
            If not None, then we only keep neurons with integrated grads in this percentile of all integrated grads.
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """

        attribution_scores, entity_ids = self.get_scores(
            entity,
            prompt,
            ground_truth,
            batch_size=batch_size,
            steps=steps,
            pbar=pbar,
            attribution_method=attribution_method,
        )
        assert (
            sum(e is not None for e in [threshold, adaptive_threshold, percentile]) == 1
        ), f"Provide one and only one of threshold / adaptive_threshold / percentile"
        if adaptive_threshold is not None:
            threshold = attribution_scores.max().item() * adaptive_threshold
        if threshold is not None:
            coarse_neurons_tmp = torch.nonzero(attribution_scores > threshold).cpu().tolist()
        else:
            s = attribution_scores.flatten().detach().cpu().numpy()
            coarse_neurons_tmp = (
                torch.nonzero(attribution_scores > np.percentile(s, percentile))
                .cpu()
                .tolist()
            )
        return [[entity_ids[x[0]], x[1]] for x in coarse_neurons_tmp]

    def get_refined_neurons(
        self,
        entity: str,
        prompts: List[str],
        ground_truth: str,
        negative_examples: Optional[List[str]] = None,
        p: float = 0.5,
        batch_size: int = 10,
        steps: int = 20,
        coarse_adaptive_threshold: Optional[float] = None,
        coarse_threshold: Optional[float] = None,
        coarse_percentile: Optional[float] = None,
        quiet=False,
    ):
        """
        Finds the 'refined' neurons for a given set of prompts and a ground truth / expected output.

        The input should be n different prompts, each expressing the same fact in different ways.
        For each prompt, we calculate the attribution scores of each intermediate neuron.
        We then set an attribution score threshold, and we keep the neurons that are above this threshold.
        Finally, considering the coarse neurons from all prompts, we set a sharing percentage threshold, p,
        and retain only neurons shared by more than p% of prompts.

        `prompts`: list of str
            the prompts to get the refined neurons for
        `ground_truth`: str
            the ground truth / expected output
        `negative_examples`: list of str
            Optionally provide a list of negative examples. Any neuron that appears in these examples will be excluded from the final results.
        `p`: float
            the threshold for the sharing percentage
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `coarse_threshold`: float
            threshold for the coarse neurons
        `coarse_percentile`: float
            percentile for the coarse neurons
        """
        assert isinstance(
            prompts, list
        ), "Must provide a list of different prompts to get refined neurons"
        assert 0.0 <= p < 1.0, "p should be a float between 0 and 1"

        n_prompts = len(prompts)
        coarse_neurons = []
        for prompt in tqdm(
            prompts, desc="Getting coarse neurons for each prompt...", disable=quiet
        ):
            coarse_neurons.append(
                self.get_coarse_neurons(
                    entity,
                    prompt,
                    ground_truth,
                    batch_size=batch_size,
                    steps=steps,
                    adaptive_threshold=coarse_adaptive_threshold,
                    threshold=coarse_threshold,
                    percentile=coarse_percentile,
                    pbar=False,
                )
            )
        if negative_examples is not None:
            negative_neurons = []
            for negative_example in tqdm(
                negative_examples,
                desc="Getting coarse neurons for negative examples",
                disable=quiet,
            ):
                negative_neurons.append(
                    self.get_coarse_neurons(
                        negative_example,
                        ground_truth,
                        batch_size=batch_size,
                        steps=steps,
                        adaptive_threshold=coarse_adaptive_threshold,
                        threshold=coarse_threshold,
                        percentile=coarse_percentile,
                        pbar=False,
                    )
                )
        # if not quiet:
        #     total_coarse_neurons = sum([len(i) for i in coarse_neurons])
        #     print(f"{total_coarse_neurons} coarse neurons found - refining")
        t = n_prompts * p
        refined_neurons = []
        c = collections.Counter()
        for neurons in coarse_neurons:
            for n in neurons:
                c[tuple(n)] += 1

        for neuron, count in c.items():
            if count > t:
                refined_neurons.append(list(neuron))

        # filter out neurons that are in the negative examples
        if negative_examples is not None:
            for neuron in negative_neurons:
                if neuron in refined_neurons:
                    refined_neurons.remove(neuron)

        total_refined_neurons = len(refined_neurons)
        if not quiet:
            print(f"{total_refined_neurons} neurons remaining after refining")
        return refined_neurons
    def get_scores_for_embd(
        self,
        token_position: int,
        prompt: str,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        encoded_input: Optional[int] = None,
        attribution_method: str = "integrated_grads",
    ):
        """
        get the attribution scores for a given layer
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `encoded_input`: int
            if not None, then use this encoded input instead of getting a new one
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        assert steps % batch_size == 0
        n_batches = steps // batch_size

        # First we take the unmodified model and use a hook to return the baseline intermediate activations at our chosen target layer
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth, encoded_input
        )
        
        # for autoregressive models, we might want to generate > 1 token
        n_sampling_steps = len(target_label)


        if attribution_method == "integrated_grads":
            integrated_grads = []

            for i in range(n_sampling_steps):
                if i > 0:
                    # retokenize new inputs
                    encoded_input, mask_idx, target_label = self._prepare_inputs(
                        prompt, ground_truth
                    )

                baseline_outputs, baseline_activations = self.get_baseline_with_activations(token_position, encoded_input)
                if n_sampling_steps > 1:
                    argmax_next_token = (
                        baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                    )
                    next_token_str = self.tokenizer.decode(argmax_next_token)

                # Now we want to gradually change the intermediate activations of our layer from 0 -> their original value
                # and calculate the integrated gradient of the masked position at each step
                # we do this by repeating the input across the batch dimension, multiplying the first batch by 0, the second by 0.1, etc., until we reach 1
                scaled_weights = self.scaled_input(
                    baseline_activations, steps=steps, device=self.device
                )
                scaled_weights.requires_grad_(True)

                integrated_grads_this_step = []  # to store the integrated gradients

                for batch_weights in scaled_weights.chunk(n_batches):
                    # we want to replace the intermediate activations at some layer, at the mask position, with `batch_weights`
                    # first tile the inputs to the correct batch size
                    inputs = {
                        "input_ids": einops.repeat(
                            encoded_input["input_ids"], "b d -> (r b) d", r=batch_size
                        ),
                        "attention_mask": einops.repeat(
                            encoded_input["attention_mask"],
                            "b d -> (r b) d",
                            r=batch_size,
                        ),
                    }
                    if self.model_type == "bert":
                        inputs["token_type_ids"] = einops.repeat(
                            encoded_input["token_type_ids"],
                            "b d -> (r b) d",
                            r=batch_size,
                        )

                    # then patch the model to replace the activations with the scaled activations
                    patch_embd(
                        self.model,
                        mask_idx=token_position,
                        embd_attr=self.embd_attr,
                        replacement_activations=batch_weights,
                    )

                    # then forward through the model to get the logits
                    outputs = self.model(**inputs)

                    # then calculate the gradients for each step w/r/t the inputs
                    probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
                    if n_sampling_steps > 1:
                        target_idx = target_label[i]
                    else:
                        target_idx = target_label
                    # torch.autograd.set_detect_anomaly(True)   # This code will seriously hurt inference speed
                    grad = torch.autograd.grad(
                        torch.unbind(probs[:, target_idx]), batch_weights
                    )[0]
                    grad = grad.sum(dim=0)
                    integrated_grads_this_step.append(grad)

                    unpatch_embed(self.model, embd_attr=self.embd_attr)

                # then sum, and multiply by W-hat / m
                integrated_grads_this_step = torch.stack(
                    integrated_grads_this_step, dim=0
                ).sum(dim=0)
                integrated_grads_this_step *= baseline_activations.squeeze(0) / steps
                integrated_grads.append(integrated_grads_this_step)

                if n_sampling_steps > 1:
                    prompt += next_token_str
            integrated_grads = torch.stack(integrated_grads, dim=0).sum(dim=0) / len(
                integrated_grads
            )
            return integrated_grads
        elif attribution_method == "max_activations":
            activations = []
            for i in range(n_sampling_steps):
                if i > 0:
                    # retokenize new inputs
                    encoded_input, mask_idx, target_label = self._prepare_inputs(
                        prompt, ground_truth
                    )
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    token_position, encoded_input
                )
                activations.append(baseline_activations)
                if n_sampling_steps > 1:
                    argmax_next_token = (
                        baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                    )
                    next_token_str = self.tokenizer.decode(argmax_next_token)
                    prompt += next_token_str
            activations = torch.stack(activations, dim=0).sum(dim=0) / len(activations)
            return activations.squeeze(0)
        else:
            raise NotImplementedError
        
    def modify_activations(
        self,
        entity: str,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        mode: str = "suppress",
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        results_dict = {}
        _, mask_idx, _ = self._prepare_inputs(
            prompt, ground_truth
        )  # just need to get the mask index for later - probably a better way to do this
        # get the baseline probabilities of the groundtruth being generated + the argmax / greedy completion before modifying the activations
        (
            gt_baseline_prob,
            argmax_baseline_prob,
            argmax_completion_str,
            _,
        ) = self._generate(prompt, ground_truth)
        if not quiet:
            print(
                f"\nBefore modification - groundtruth probability: {gt_baseline_prob}\nArgmax completion: `{argmax_completion_str}`\nArgmax prob: {argmax_baseline_prob}\n"
            )
        results_dict["before"] = {
            "gt_prob": gt_baseline_prob,
            "argmax_completion": argmax_completion_str,
            "argmax_prob": argmax_baseline_prob,
        }

        # patch model to suppress neurons
        # store all the layers we patch so we can unpatch them later
        _, positions, _ = self._prepare_ids(entity, prompt)
        patch_embd(
            self.model,
            mask_idx,
            mode=mode,
            neurons=neurons,
            embd_attr=self.embd_attr,
            multi_mask_idx=positions
        )

        # get the probabilities of the groundtruth being generated + the argmax / greedy completion after modifying the activations
        new_gt_prob, new_argmax_prob, new_argmax_completion_str, _ = self._generate(
            prompt, ground_truth
        )
        if not quiet:
            print(
                f"\nAfter modification - groundtruth probability: {new_gt_prob}\nArgmax completion: `{new_argmax_completion_str}`\nArgmax prob: {new_argmax_prob}\n"
            )
        results_dict["after"] = {
            "gt_prob": new_gt_prob,
            "argmax_completion": new_argmax_completion_str,
            "argmax_prob": new_argmax_prob,
        }

        unpatch_embed(self.model, self.embd_attr)

        return results_dict

    def suppress_knowledge(
        self,
        entity: str,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        undo_modification: bool = True,
        quiet: bool = False
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, zeroing the activations at the positions specified by `neurons`,
        and measure the resulting affect on the ground truth probability.
        """
        return self.modify_activations(
            entity=entity,
            prompt=prompt,
            ground_truth=ground_truth,
            neurons=neurons,
            mode="suppress",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    def enhance_knowledge(
        self,
        entity: str,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, multiplying the activations at the positions
        specified by `neurons` by 2, and measure the resulting affect on the ground truth probability.
        """
        return self.modify_activations(
            entity=entity,
            prompt=prompt,
            ground_truth=ground_truth,
            neurons=neurons,
            mode="enhance",
            undo_modification=undo_modification,
            quiet=quiet,
        )
    



def model_type(model_name: str):
    if "gpt2" in model_name:
        return "gpt2"
    elif "gpt_neo" in model_name:
        return "gpt_neo"
    elif "llama" in model_name:
        return "llama"
    elif "phi" in model_name:
        return "phi"
    elif "gpt-j" in model_name:
        return "gpt-j"
    else:
        raise ValueError("Model {model_name} not supported")