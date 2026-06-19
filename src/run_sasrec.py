import sys

import numpy as np

if not hasattr(np, "float"):
    np.float = float

import torch


original_torch_load = torch.load


def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return original_torch_load(*args, **kwargs)


torch.load = torch_load_compat

from recbole.quick_start import run_recbole


config_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "configs/sasrec_ce_ml1m.yaml"
)

run_recbole(
    model="SASRec",
    dataset="ml-1m",
    config_file_list=[config_path],
)