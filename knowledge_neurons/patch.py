# helper functions for patching torch transformer models
import torch
import torch.nn as nn
import collections
from typing import List, Callable
import torch
import torch.nn.functional as F
import collections


def get_attributes(x: nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.encoder.layer')
        should return the same as model.transformer.encoder.layer
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x


def set_attribute_recursive(x: nn.Module, attributes: "str", new_attribute: nn.Module):
    """
    Given a list of period-separated attributes - set the final attribute in that list to the new value
    i.e set_attribute_recursive(model, 'transformer.encoder.layer', NewLayer)
        should set the final attribute of model.transformer.encoder.layer to NewLayer
    """
    for attr in attributes.split(".")[:-1]:
        x = getattr(x, attr)
    setattr(x, attributes.split(".")[-1], new_attribute)


class Patch(torch.nn.Module):
    """
    Patches a torch module to replace/suppress/enhance the intermediate activations
    """

    def __init__(
        self,
        layer: nn.Module,
        mask_idx: int,
        replacement_activations: torch.Tensor = None,
        target_positions: List[List[int]] = None,
        mode: str = "replace",
        enhance_value: float = 2.0,
        multi_mask_idx: List[int] = None
    ):
        super().__init__()
        self.original_layer = layer
        self.acts = replacement_activations
        self.mask_idx = mask_idx
        self.target_positions = target_positions
        self.enhance_value = enhance_value
        assert mode in ["replace", "suppress", "enhance"]
        self.mode = mode
        self.multi_mask_idx = multi_mask_idx
        if self.mode == "replace":
            assert self.acts is not None
        elif self.mode in ["enhance", "suppress"]:
            assert self.target_positions is not None

    def forward(self, x: torch.Tensor):
        x = self.original_layer(x)
        if self.mode == "replace":
            x[:, self.mask_idx, :] = self.acts
        elif self.mode == "suppress":
            if self.multi_mask_idx:
              for _id in self.multi_mask_idx:
                for pos in self.target_positions:
                    x[:, _id, pos] = 0.0
        elif self.mode == "enhance":
            if self.multi_mask_idx:
              for _id in self.multi_mask_idx:
                for pos in self.target_positions:
                    x[:, _id, pos] *= self.enhance_value
        else:
            raise NotImplementedError
        return x

def patch_embd(
    model: nn.Module,
    mask_idx: int,
    embd_attr: str,
    replacement_activations: torch.Tensor = None,
    mode: str = "replace",
    neurons: List[List[int]] = None,
    multi_mask_idx: List[int] = None
):
    """
    replaces the ff layer at `layer_idx` with a `Patch` class - that will replace the intermediate activations at sequence position
    `mask_index` with `replacement_activations`

    `model`: nn.Module
      a torch.nn.Module [currently only works with HF Bert models]
    `mask_idx`: int
      the index (along the sequence length) of the activation to replace.
      TODO: multiple indices
    `ff_attrs`: str
      chain of attributes (separated by periods) that access the ff block within a transformer layer
    """
    embedding_layer = model.get_input_embeddings()
    if mode == "replace":
        set_attribute_recursive(model, embd_attr, Patch(
                embedding_layer,
                mask_idx,
                replacement_activations=replacement_activations,
                mode=mode,
            ))
        assert isinstance(model.get_input_embeddings(), Patch)
    elif mode in ["suppress", "enhance"]:
        neurons_dict = collections.defaultdict(list)
        for neuron in neurons:
            token_idx, pos = neuron
            neurons_dict[token_idx].append(pos)
        for token_idx, positions in neurons_dict.items():
            set_attribute_recursive(model, embd_attr, Patch(
                embedding_layer,
                token_idx,
                replacement_activations=None,
                mode=mode,
                target_positions=positions,
                multi_mask_idx = multi_mask_idx
            ))
    else:
        raise NotImplementedError


def unpatch_embed(
    model: nn.Module,
    embd_attr: str,
):
    """
    Removes the `Patch` applied by `patch_ff_layer`, replacing it with its original value.

    `model`: torch.nn.Module
      a torch.nn.Module [currently only works with HF Bert models]
    `layer_idx`: int
      which transformer layer to access
    `transformer_layers_attr`: str
      chain of attributes (separated by periods) that access the transformer layers within `model`.
      The transformer layers are expected to be indexable - i.e a Modulelist
    `ff_attrs`: str
      chain of attributes (separated by periods) that access the ff block within a transformer layer
    """
    embedding_layer = model.get_input_embeddings()
    assert isinstance(embedding_layer, Patch), "Can't unpatch a layer that hasn't been patched"
    set_attribute_recursive(model, embd_attr, embedding_layer.original_layer)

  