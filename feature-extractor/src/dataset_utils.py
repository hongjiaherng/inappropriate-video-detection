import glob
import os
from pathlib import Path
from typing import Dict, List, Set, Optional
from urllib.parse import urlsplit

import datasets


def split_filepath(full_filepath: str, base_dirpath: str) -> Dict:
    full_filepath = Path(full_filepath)
    return {
        "filename": full_filepath.stem,  # filename without ext
        "relative_dir": full_filepath.parent.relative_to(
            base_dirpath
        ).as_posix(),  # relative dir to base_dirpath
    }


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
                "multilabel_target",
                "frame_annotations",
            ]
        )  # Remove unused columns for preprocessing

        test_ds = datasets.load_dataset(
            hf_dataset_name, name="video", split="test", streaming=True
        ).map(
            remove_columns=[
                "binary_target",
                "multilabel_target",
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
