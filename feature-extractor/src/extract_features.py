import os
import shutil
import gc
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
import tqdm.auto as tqdm


def forward_batch(
    batch: torch.Tensor, model: nn.Module, device: torch.device
) -> np.ndarray:
    with torch.no_grad():
        batch = batch.to(device)  # (batch_size, C, T, H, W)
        batch = model(batch).cpu().numpy()

    return batch  # (batch_size, 2042)


def extract_one_off(
    video_dict: Dict,
    model: nn.Module,
    preprocessing: Union[Dict[str, nn.Module], nn.Module],
    device: torch.device,
    temp_dir: str,
) -> np.ndarray:
    """
    video_dict: Dict[str, str] = {"id": str, "path": str}

    """
    video_in = preprocessing(video_dict)["inputs"]  # (N, n_crops, C, T, H, W)
    num_clips = video_in.size()[0]

    for clip_idx in tqdm.tqdm(
        range(num_clips),
        desc=f'Processing clips of "{video_dict["id"]}"',
        unit="clip",
        position=1,
        leave=False,
        colour="green",
    ):
        # treat n crops as a batch
        batch_in = video_in[clip_idx]  # (n_crops, C, T, H, W)
        batch_out = forward_batch(batch_in, model, device)  # (n_crops, 2042)

        np.save(os.path.join(temp_dir, f"{video_dict['id']}_{clip_idx}.npy"), batch_out)

        del batch_in, batch_out
        gc.collect()

    del video_in
    gc.collect()

    # combine all clip features of a video into a single tensor (video features)
    video_emb = np.stack(
        [
            np.load(os.path.join(temp_dir, f"{video_dict['id']}_{clip_idx}.npy"))
            for clip_idx in range(num_clips)
        ],
        axis=0,
    )  # list of (n_crops, 2042) -> (n_clips, n_crops, 2042)

    return video_emb


def extract_by_batches(
    video_dict: Dict,
    model: nn.Module,
    preprocessing: Union[Dict[str, nn.Module], nn.Module],
    device: torch.device,
    temp_dir: str,
) -> np.ndarray:
    """
    video_dict: Dict[str, str] = {"id": str, "path": str}
    """
    batch_iter = preprocessing["video"](video_dict)
    num_batches = len(batch_iter)

    for batch_idx, batch_dict in tqdm.tqdm(
        enumerate(batch_iter),
        total=num_batches,
        desc=f'Processing clips of "{video_dict["id"]}"',
        unit="batch of clips",
        position=1,
        leave=False,
        colour="green",
    ):
        # {"inputs": torch.Size([B, T, C, H, W]), "meta": Dict[str, Any]} -> {"inputs": torch.Size([B, n_crops, C, T, H, W]), "meta": Dict[str, Any]}
        batch_dict = preprocessing["clip"](batch_dict)
        batch_in = torch.permute(
            batch_dict["inputs"], (1, 0, 2, 3, 4, 5)
        )  # (B, n_crops, C, T, H, W) -> (n_crops, B, C, T, H, W)
        batch_out = []

        for crop_idx in range(batch_in.size()[0]):
            batch_out.append(
                forward_batch(batch_in[crop_idx], model, device)  # (B, 2042)
            )

        batch_out = np.stack(batch_out, axis=1)  # (B, n_crops, 2042)

        np.save(
            os.path.join(temp_dir, f"{video_dict['id']}_{batch_idx}.npy"), batch_out
        )

        del batch_in, batch_out, batch_dict
        gc.collect()

    del batch_iter
    gc.collect()

    # combine all clip features of a video into a single tensor (video features)
    video_emb = np.concatenate(
        [
            np.load(os.path.join(temp_dir, f"{video_dict['id']}_{batch_idx}.npy"))
            for batch_idx in range(num_batches)
        ],
        axis=0,
    )  # (N, n_crops, 2042)

    return video_emb


def extract_features(
    video_dict: Dict,
    model: nn.Module,
    preprocessing: Union[Dict[str, nn.Module], nn.Module],
    device: torch.device,
    temp_dir: str,
) -> np.ndarray:
    try:
        os.makedirs(temp_dir, exist_ok=True)

        if isinstance(preprocessing, dict):
            video_emb = extract_by_batches(
                video_dict, model, preprocessing, device, temp_dir
            )
        else:
            video_emb = extract_one_off(
                video_dict, model, preprocessing, device, temp_dir
            )
    finally:
        shutil.rmtree(temp_dir)

    return video_emb
