import torch
import numpy as np


def get_actual_seq_len(xs):
    return torch.sum(torch.max(torch.abs(xs), dim=2)[0] > 0, dim=1)


def run(model, device):
    xs = torch.from_numpy(np.load("xs.npy"))
    ys = torch.from_numpy(np.load("ys.npy"))

    seq_len = get_actual_seq_len(xs)

    # Shrink the sequence length as much as possible to the maximum sequence length in the batch
    xs = xs[:, : torch.max(seq_len), :]  # (B, T, D)
    print(f"{xs.shape=}")
    xs = xs.to(device)
    ys = ys.to(device)

    for k, v in model.named_parameters():
        print(f"{k}: {v.shape}")

    outs = model(xs, seq_len)

    print(model)
