import argparse
import distutils.util
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: Optional[int]):
    # Ensure deterministic behavior
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def int_or_none(value):
    if value.lower() == "none":
        return None
    return int(value)


def str_or_none(value):
    if value.lower() == "none":
        return None
    return str(value)


def str2bool(value):
    try:
        return bool(distutils.util.strtobool(value))
    except Exception:
        raise argparse.ArgumentTypeError("Boolean value expected.")
