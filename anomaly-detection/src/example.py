import random
import torch
import torch.utils.data
import numpy as np
import dataset
import utils


if __name__ == "__main__":
    seed = 42
    utils.seed_everything(seed)

    train_loader = dataset.train_loader(
        "i3d_rgb",
        batch_size=128,
        max_seq_len=100,
        streaming=True,
        num_workers=4,
        shuffle=True,
        seed=seed,
    )
    print(train_loader)

    for i, x in enumerate(train_loader):
        if i == 50:
            break
        print(f'{i=}; {x["id"]=}; {x["binary_target"]=}; {x["feature"].shape=}')

