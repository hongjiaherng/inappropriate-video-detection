import os
import shutil
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm.auto as tqdm

torchvision.disable_beta_transforms_warning()


def forward_batch(
    batch: torch.Tensor, model: nn.Module, device: torch.device
) -> np.ndarray:
    with torch.no_grad():
        batch = batch.to(device)  # (batch_size, C, T, H, W)
        batch = model(batch).detach().cpu().numpy()

    return batch  # (batch_size, 2042)


def extract_features(
    video_ex: Dict,
    model: nn.Module,
    preprocessing: Union[Dict[str, nn.Module], nn.Module],
    device: torch.device,
    temp_dir: str,
) -> np.ndarray:
    os.makedirs(temp_dir, exist_ok=True)
    video_id = video_ex["id"]

    if isinstance(preprocessing, dict):
        # a list of unrealized tensor of (batch_size, n_crops, C, T, H, W)
        batches = preprocessing["video2clips"](video_ex)
        num_batches = len(batches)

        for batch_idx in tqdm.tqdm(
            range(num_batches),
            desc=f'Processing clips of "{video_id}"',
            unit="batch of clips",
            position=1,
            leave=False,
            colour="green",
        ):
            # (batch_size, n_crops, C, T, H, W)
            batch = preprocessing["clip_preprocessing"](batches[batch_idx])["inputs"]
            # Putting n_crops as first dimension
            batch = torch.permute(batch, (1, 0, 2, 3, 4, 5))
            batch_emb = []

            # extract crop-by-crop
            for crop in range(batch.size()[0]):
                # (batch_size, C, T, H, W) -> (batch_size, 2042)
                batch_crop = forward_batch(batch[crop], model, device)
                batch_emb.append(batch_crop)

            # combine list of batch_crop into a single batch_emb
            # list of (batch_size, 2042) -> (batch_size, n_crops, 2042)
            batch_emb = np.stack(batch_emb, axis=1)

            np.save(os.path.join(temp_dir, f"{video_id}_{batch_idx}.npy"), batch_emb)

            # free up memory
            del batch, batch_emb, batch_crop

        # combine all clip features of a video into a single tensor (video features)
        # list of (batch_size, n_crops, 2042) -> (n_clips, n_crops, 2042)
        video_emb = np.concatenate(
            [
                np.load(os.path.join(temp_dir, f"{video_id}_{batch_idx}.npy"))
                for batch_idx in range(num_batches)
            ],
            axis=0,
        )

        del batches

    else:
        # realized tensor of (num_clips, n_crops, C, T, H, W), loaded into memory
        video_ex = preprocessing(video_ex)
        num_clips = video_ex["inputs"].size()[0]

        for clip_idx in tqdm.tqdm(
            range(num_clips),
            desc=f'Processing clips of "{video_id}"',
            unit="clip",
            position=1,
            leave=False,
            colour="green",
        ):
            # treat n crops as a batch
            batch = video_ex["inputs"][clip_idx]  # (n_crops, C, T, H, W)
            batch_emb = forward_batch(batch, model, device)  # (n_crops, 2042)

            np.save(os.path.join(temp_dir, f"{video_id}_{clip_idx}.npy"), batch_emb)

            del batch, batch_emb

        # combine all clip features of a video into a single tensor (video features)
        video_emb = np.stack(
            [
                np.load(os.path.join(temp_dir, f"{video_id}_{clip_idx}.npy"))
                for clip_idx in range(num_clips)
            ],
            axis=0,
        )  # list of (n_crops, 2042) -> (n_clips, n_crops, 2042)

    shutil.rmtree(temp_dir)

    return video_emb
