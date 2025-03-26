from dataclasses import dataclass
from typing import List
from utils.hparams import HyperParams


@dataclass
class GraceHyperParams(HyperParams):
    # Experiments
    
    edit_lr: int
    n_iter: int
    # Method
    eps: float
    dist_fn: str
    val_init: str
    val_train: str
    val_reg: str
    reg: str
    replacement: str
    eps_expand: str
    num_pert: str
    dropout: float

    # Module templates
    inner_params: List[str]
