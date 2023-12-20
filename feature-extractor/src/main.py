import argparse
import gc
import importlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm.auto as tqdm
import dataset_utils
from extract_features import extract_features


ROOT_PATH = os.path.abspath(os.path.join(__file__, "../.."))

torchvision.disable_beta_transforms_warning()

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(args):
    # Get configs from args
    input_dir = os.path.abspath(args.input_dir) if args.input_dir else None
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else None
    temp_dir = os.path.join(output_dir, "temp")

    batch_size = int(args.batch_size)
    hf_dataset_name = args.hf_dataset
    model_name = args.model
    num_clips_per_video = int(args.num_clips_per_video)
    crop_type = args.crop_type
    io_backend = "http" if hf_dataset_name else "local"
    force_stop_after_n_videos = args.force_stop_after_n_videos
    check_progress = (
        os.path.abspath(args.check_progress) if args.check_progress else None
    )

    logger.info(
        f"""Configs: {json.dumps(
        {
            "batch_size": batch_size,
            "hf_dataset": hf_dataset_name,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "temp_dir": temp_dir,
            "model": model_name,
            "num_clips_per_video": num_clips_per_video,
            "crop_type": crop_type,
            "io_backend": io_backend,
            "force_stop_after_n_videos": force_stop_after_n_videos,
            "check_progress": check_progress
        },
        indent=2,
    )}"""
    )

    # Check current progress
    progress = dataset_utils.get_progress(output_dir, check_progress)
    logger.info(f"Current progress: {len(progress)} done")

    # Init dataset that excludes already done progress
    if hf_dataset_name:
        ds, extract_relative_dir = dataset_utils.init_hf_dataset(
            hf_dataset_name, progress
        )
    else:
        ds, extract_relative_dir = dataset_utils.init_local_dataset(input_dir, progress)

    logger.info(f"Dataset: {'huggingface' if hf_dataset_name else 'local'}")

    # Init model and preprocessing pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocessing = init_model(
        model_name=model_name,
        io_backend=io_backend,
        num_clips_per_video=num_clips_per_video,
        crop_type=crop_type,
        batch_size=batch_size,
    )
    model.to(device)

    logger.info(f"Model: {model_name}")
    logger.info(f"Preprocessing pipeline: {preprocessing}")
    logger.info(f"Device: {device}")

    for i, video_dict in enumerate(
        tqdm.tqdm(
            ds,
            desc="Extracting video features",
            unit="video",
            total=None if hf_dataset_name else len(ds),
            position=0,
            leave=True,
        )
    ):
        if i == force_stop_after_n_videos:
            tqdm.tqdm.write(
                f"INFO:{__name__}:Force stop after {force_stop_after_n_videos} videos"
            )
            break

        video_reldir = extract_relative_dir(video_dict["path"])
        video_out_dir = os.path.join(output_dir, video_reldir)

        # Extract features of all clips in the video and concat them into one as a video feature
        video_emb = extract_features(
            video_dict=video_dict,
            model=model,
            preprocessing=preprocessing,
            device=device,
            temp_dir=temp_dir,
        )

        # save video features to output_dir
        # create video output dir if havent exist
        os.makedirs(video_out_dir, exist_ok=True)
        np.save(os.path.join(video_out_dir, f"{video_dict['id']}.npy"), video_emb)

        if check_progress:
            with open(check_progress, mode="a", encoding="utf-8") as f:
                f.write(
                    f"{Path(os.path.join(video_reldir, video_dict['id'])).as_posix()}\n"
                )

        # free up memory
        del video_dict, video_emb, video_reldir, video_out_dir
        gc.collect()


def init_model(
    model_name: str,
    io_backend: str,
    num_clips_per_video: int,
    crop_type: str,
    batch_size: int,
) -> Tuple[nn.Module, Union[Dict[str, nn.Module], nn.Module]]:
    """
    Initialize model and preprocessing pipeline.
    """

    if (
        model_name
        == "i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb"
    ):
        i3d = importlib.import_module(
            "models.i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb"
        )
        model = i3d.build_model()

        if batch_size == -1:
            preprocessing = i3d.build_end2end_pipeline(
                io_backend=io_backend,
                num_clips=num_clips_per_video,
                crop_type=crop_type,
            )
        else:
            preprocessing = {
                "video": i3d.build_video2clips_pipeline(
                    batch_size=batch_size,
                    io_backend=io_backend,
                    num_clips=num_clips_per_video,
                ),
                "clip": i3d.build_clip_pipeline(crop_type=crop_type),
            }

    elif (
        model_name == "swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb"
    ):
        swin = importlib.import_module(
            "models.swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb"
        )
        model = swin.build_model()

        if batch_size == -1:
            preprocessing = swin.build_end2end_pipeline(
                io_backend=io_backend,
                num_clips=num_clips_per_video,
                crop_type=crop_type,
            )
        else:
            preprocessing = {
                "video": swin.build_video2clips_pipeline(
                    batch_size=batch_size,
                    io_backend=io_backend,
                    num_clips=num_clips_per_video,
                ),
                "clip": swin.build_clip_pipeline(crop_type=crop_type),
            }
    else:
        raise ValueError(
            f"Model {model_name} not supported. Currently only supports ['i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb', 'swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb']."
        )

    return model, preprocessing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--hf_dataset",
        type=str,
        help='HuggingFace dataset name that contains source videos to extract visual features from. Currently only supports ["jherng/xd-violence"].',
        choices=["jherng/xd-violence"],
    )
    group.add_argument(
        "--input_dir",
        type=str,
        help="Directory for input videos if you wish to extract visual features of videos from local disk (Optional).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(ROOT_PATH, "data", "outputs", "feature"),
        help="Directory for output features.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for clips. 1 batch ~ 32 MB (batch_size * n_crops * C * T * H * W * 4 bytes = 1 * 5 * 3 * 32 * 224 * 224 * 4 / 1e6 = 32 MB)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb",
        help='Model name to use for feature extraction. Currently only supports ["i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb"].',
        choices=[
            "i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb",
            "swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb",
        ],
    )
    parser.add_argument(
        "--num_clips_per_video",
        type=int,
        default=-1,
        help="Number of clips to sample from each video. If -1, then all possible clips will be sampled from start to end (must statisfy (== -1 || >= 1)).",
    )
    parser.add_argument(
        "--crop_type",
        type=str,
        default="5-crop",
        help='Type of cropping technique to use for cutting input video spatially into n cropped video of size 224x224 before feature extraction. Currently only supports ["10-crop", "5-crop", "center"].',
        choices=["10-crop", "5-crop", "center"],
    )
    parser.add_argument(
        "--force_stop_after_n_videos",
        type=int,
        default=-1,
        help="The number of videos to extract feature from. If -1, all the videos in the dataset will be extracted exhaustively.",
    )
    parser.add_argument(
        "--check_progress",
        type=str,
        default=None,
        help="How to determine the progress of feature extraction? If None, check <output_dir>, if <filename>, check line by line.",
    )
    args = parser.parse_args()

    if not (int(args.num_clips_per_video) == -1 or int(args.num_clips_per_video) >= 1):
        parser.error("--num_clips_per_video must statisfy (== -1 || >= 1)")

    if not (
        int(args.force_stop_after_n_videos) == -1
        or int(args.force_stop_after_n_videos) >= 1
    ):
        parser.error("--force_stop_after_n_videos must statisfy (== -1 || >= 1)")

    run(args)
