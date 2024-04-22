from dataclasses import dataclass
from typing import List
from typing_extensions import Literal

from utils.hparams import HyperParams


@dataclass
class SWEAOSHyperParams(HyperParams):
    # Method
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer_name: str
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    nll_loss_factor: float


    # Module templates
    embedding_module: str
    ln_f_module: str
    lm_head_module: str
    mode: str
    kn_hparams: list
    #cat: coarse_adaptive_threshold, ct: coarse_threshold, cp: coarse_percentile
