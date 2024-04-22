from dataclasses import dataclass

from utils.hparams import HyperParams


@dataclass
class MENDHyperParams(HyperParams):
    lr_scale: float
    n_toks: int
    model_name: str
    counterfact: bool
    mini: bool
    zsre: bool
