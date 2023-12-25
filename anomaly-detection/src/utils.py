import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: Optional[int]):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
