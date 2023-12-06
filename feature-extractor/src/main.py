# python src/main.py --hf_dataset="jherng/xd-violence" --output_dir="data/outputs/i3d_rgb" --model="i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb" --num_clips_per_video=-1 --crop_type="5-crop"

import argparse
import gc
import glob
import importlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from urllib.parse import urlsplit

import datasets
import numpy as np
import torch
import torch.nn as nn
import tqdm
import torchvision

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
    progress = get_progress(output_dir, check_progress)
    logger.info(f"Current progress: {len(progress)} done")

    # Init dataset that excludes already done progress
    if hf_dataset_name:
        ds, extract_relative_dir = init_hf_dataset(hf_dataset_name, progress)
    else:
        ds, extract_relative_dir = init_local_dataset(input_dir, progress)

    logger.info(f"Dataset: {'huggingface' if hf_dataset_name else 'local'}")

    # Init model and preprocessing pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocessing = init_model(
        model_name=model_name,
        io_backend=io_backend,
        num_clips_per_video=num_clips_per_video,
        crop_type=crop_type,
    )
    model.to(device)

    logger.info(f"Model: {model_name}")
    logger.info(f"Preprocessing pipeline: {preprocessing}")
    logger.info(f"Device: {device}")

    with torch.no_grad():
        for i, video_ex in enumerate(
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
                print()
                logger.info(f"Force stop after {force_stop_after_n_videos} videos")
                break

            video_ex = preprocessing(video_ex)

            # garbage collect any possible unused obj
            gc.collect()

            # iterate over each clip in the video (clip dimension)
            num_clips_cur_video = video_ex["inputs"].size()[0]
            os.makedirs(temp_dir, exist_ok=True)
            for clip_idx in tqdm.tqdm(
                range(num_clips_cur_video),
                desc=f"Processing clips of \"{video_ex['meta']['id']}\"",
                unit="clip",
                position=1,
                leave=False,
                colour="green",
            ):
                # treat n crops as a batch
                batch_clip = video_ex["inputs"][clip_idx]  # (n_crops, C, T, H, W)
                batch_clip = batch_clip.to(device)

                # extract features
                batch_emb = model(batch_clip)

                # save a clip feature to temp_dir
                batch_emb = batch_emb.detach().cpu().numpy()
                np.save(
                    os.path.join(temp_dir, f"{video_ex['meta']['id']}_{clip_idx}.npy"),
                    batch_emb,
                )

                # free up memory
                del batch_clip, batch_emb

            # combine all clip features of a video into a single tensor (video features)
            video_emb = np.stack(
                [
                    np.load(
                        os.path.join(
                            temp_dir, f"{video_ex['meta']['id']}_{clip_idx}.npy"
                        )
                    )
                    for clip_idx in range(num_clips_cur_video)
                ],
                axis=0,
            )

            # save video features to output_dir
            # create video output dir if havent exist
            video_output_dir = os.path.join(
                output_dir, extract_relative_dir(video_ex["meta"]["filename"])
            )
            os.makedirs(video_output_dir, exist_ok=True)
            np.save(
                os.path.join(video_output_dir, f"{video_ex['meta']['id']}.npy"),
                video_emb,
            )
            if check_progress:
                with open(check_progress, mode="a", encoding="utf-8") as f:
                    f.write(
                        f"""{Path(
                            os.path.join(
                                extract_relative_dir(video_ex["meta"]["filename"]),
                                video_ex["meta"]["id"],
                            )
                        ).as_posix()}\n"""
                    )

            # free up memory
            del video_ex, video_emb
            gc.collect()
            shutil.rmtree(temp_dir)


def get_progress(output_dir: str, check_progress: Optional[str]) -> Set[str]:
    """
    Get the progress of the feature extraction process by checking the output_dir for existing features.
    Each str in progress is in the format of <relative_dir>/<video_id>
    """

    progress = set()

    if not check_progress:
        # empty progress if output_dir doesn't even exist
        if not os.path.exists(output_dir):
            return progress

        for fp in glob.glob(os.path.join(output_dir, "**/*.npy"), recursive=True):
            splitted = split_filepath(full_filepath=fp, base_dirpath=output_dir)
            joined = Path(
                os.path.join(splitted["relative_dir"], splitted["filename"])
            ).as_posix()  # join relative dir to filename without ext (e.g., 1004-2005/this_video_name)

            progress.add(joined)
    else:
        if os.path.exists(check_progress):
            with open(check_progress, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    progress.add(line.strip())

    return progress


def split_filepath(full_filepath: str, base_dirpath: str) -> Dict:
    full_filepath = Path(full_filepath)
    return {
        "filename": full_filepath.stem,  # filename without ext
        "relative_dir": full_filepath.parent.relative_to(
            base_dirpath
        ).as_posix(),  # relative dir to base_dirpath
    }


def init_hf_dataset(
    hf_dataset_name: str, progress: Set[str]
) -> datasets.IterableDataset:
    """
    Initialize HuggingFace dataset (both train and test splits) and filter out videos that have already been processed.
    Note: Currently only supports streaming huggingface datasets but not non-streaming huggingface dataset.
    """

    if hf_dataset_name == "jherng/xd-violence":

        def extract_relative_dir(full_filepath: str):
            data_url = "/datasets/jherng/xd-violence/resolve/main/data/video"
            return "/".join(
                urlsplit(full_filepath)
                .path.split(data_url)[-1]
                .lstrip("/")
                .split("/")[:-1]  # relative_dir
            )

        train_ds = datasets.load_dataset(
            hf_dataset_name, name="video", split="train", streaming=True
        ).map(
            remove_columns=[
                "binary_target",
                "multilabel_targets",
                "frame_annotations",
            ]
        )  # Remove unused columns for preprocessing

        test_ds = datasets.load_dataset(
            hf_dataset_name, name="video", split="test", streaming=True
        ).map(
            remove_columns=[
                "binary_target",
                "multilabel_targets",
                "frame_annotations",
            ]
        )

        # Concatenate train and test datasets
        combined_ds = datasets.concatenate_datasets([train_ds, test_ds])

        # Filter out videos that have already been processed
        # assume there's always a subdir in the path at 2nd last position,
        # e.g., 1-1004 from https://huggingface.co/datasets/.../1-1004/A.Beautiful.Mind.2001__%2300-01-45_00-02-50_label_A.mp4
        combined_ds = combined_ds.filter(
            lambda x: "/".join([extract_relative_dir(x["path"]), x["id"]])
            not in progress
        )

    else:
        raise ValueError(
            f"Dataset {hf_dataset_name} not supported. Currently only supports ['jherng/xd-violence']."
        )

    return combined_ds, extract_relative_dir


def init_local_dataset(input_dir: str, progress: Set[str]) -> List[dict]:
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory '{input_dir}' does not exist.")

    def extract_relative_dir(full_filepath: str):
        return split_filepath(full_filepath=full_filepath, base_dirpath=input_dir)[
            "relative_dir"
        ]

    ds = glob.glob(os.path.join(input_dir, "**/*.mp4"), recursive=True)

    if len(ds) == 0:
        raise ValueError(f"No videos found in {input_dir}.")

    ds = [
        {
            "id": split_filepath(x, input_dir)["filename"],
            "path": x,
        }
        for x in ds
    ]

    # Filter out videos that have already been processed
    ds = list(
        filter(
            lambda x: "/".join(
                list(
                    part
                    for part in [
                        extract_relative_dir(x["path"]),
                        x["id"],
                    ]
                    if part != "."
                )  # remove empty relative dir / subdir which is denoted by "."
            )
            not in progress,
            ds,
        )
    )

    return ds, extract_relative_dir


def init_model(
    model_name: str,
    io_backend: str,
    num_clips_per_video: int,
    crop_type: str,
) -> Tuple[nn.Module, nn.Module]:
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
        preprocessing = i3d.build_preprocessing_pipeline(
            io_backend=io_backend,
            num_clips=num_clips_per_video,
            crop_type=crop_type,
        )
    else:
        raise ValueError(
            f"Model {model_name} not supported. Currently only supports ['i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb']."
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
        "--model",
        type=str,
        default="i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb",
        help='Model name to use for feature extraction. Currently only supports ["i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb"].',
        choices=[
            "i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb"
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
