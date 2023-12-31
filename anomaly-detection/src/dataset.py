from typing import Any, Dict, List, Literal, Optional

import datasets
import torch
import torch.nn
import torch.utils.data
from transforms import ExtractFrameGTs, UniformSubsampleOrPad


def test_batch_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for test dataloader

    Required keys in batch[idx]:
    ```
    * feature: torch.Tensor     # (T, D)
    * frame_gts: torch.Tensor   # (T,)
    * binary_target: torch.Tensor   # (T,)
    ```

    Resulting keys (unspecified keys are collated using default_collate):
    ```
    + feature: torch.Tensor     # (n_crops, T, D)
    + frame_gts: torch.Tensor   # (T,)
    + binary_target: torch.Tensor   # (1,)
    ```

    """
    feature = torch.stack([x["feature"] for x in batch], dim=0)
    frame_gts = batch[0]["frame_gts"]  # Take the first crop
    binary_target = batch[0]["binary_target"]  # Take the first crop

    # Stack the rest of the keys in batch if any
    remaining = {}
    remaining_k = batch[0].keys() - {"feature", "frame_gts", "binary_target"}
    if len(remaining_k) > 0:
        remaining = torch.utils.data.default_collate([{k: x[k] for k in remaining_k} for x in batch])

    return {"feature": feature, "frame_gts": frame_gts, "binary_target": binary_target, **remaining}


def streaming_test_dataset(config_name: Literal["i3d_rgb", "swin_rgb"]) -> datasets.IterableDataset:
    assert config_name in ["i3d_rgb", "swin_rgb"], "Only i3d_rgb and swin_rgb are supported for now"

    return (
        datasets.load_dataset("jherng/xd-violence", config_name, split="test", streaming=True)
        .with_format("torch")
        .map(ExtractFrameGTs(clip_len=32, sampling_rate=2))
        .remove_columns(["id", "multilabel_target", "frame_annotations"])
    )


def cached_test_dataset(config_name: Literal["i3d_rgb", "swin_rgb"], num_workers: Optional[int]) -> datasets.Dataset:
    assert config_name in ["i3d_rgb", "swin_rgb"], "Only i3d_rgb and swin_rgb are supported for now"
    assert num_workers is None or num_workers > 0, "num_workers must be > 0 or None"

    return (
        datasets.load_dataset("jherng/xd-violence", config_name, split="test", streaming=False, num_proc=num_workers)
        .with_format("torch")
        .map(ExtractFrameGTs(clip_len=32, sampling_rate=2), num_proc=num_workers)
        .remove_columns(["id", "multilabel_target", "frame_annotations"])
    )


def streaming_train_dataset(
    config_name: Literal["i3d_rgb", "swin_rgb"],
    max_seq_len: int,
    shuffle: bool,
    seed: Optional[int],
    filter: Literal["anomaly", "normal", "all"] = "all",
) -> datasets.IterableDataset:
    assert config_name in ["i3d_rgb", "swin_rgb"], "Only i3d_rgb and swin_rgb are supported for now"
    assert filter in ["anomaly", "normal", "all"], "filter must be one of anomaly, normal, all"

    ds = datasets.load_dataset("jherng/xd-violence", config_name, split="train", streaming=True).with_format("torch")

    if filter == "anomaly":
        ds = ds.filter(lambda x: x["binary_target"] == 1)
    elif filter == "normal":
        ds = ds.filter(lambda x: x["binary_target"] == 0)

    ds = ds.map(UniformSubsampleOrPad(max_seq_len=max_seq_len)).remove_columns(["id", "multilabel_target", "frame_annotations"])

    if not shuffle:
        return ds

    return ds.shuffle(seed=seed, buffer_size=50)


def cached_train_dataset(
    config_name: Literal["i3d_rgb", "swin_rgb"],
    max_seq_len: int,
    num_workers: Optional[int],
    filter: Literal["anomaly", "normal", "all"] = "all",
) -> datasets.Dataset:
    assert config_name in ["i3d_rgb", "swin_rgb"], "Only i3d_rgb and swin_rgb are supported for now"
    assert num_workers is None or num_workers > 0, "num_workers must be > 0 or None"
    assert filter in ["anomaly", "normal", "all"], "filter must be one of anomaly, normal, all"

    dataset = datasets.load_dataset("jherng/xd-violence", config_name, split="train", streaming=False, num_proc=num_workers).with_format("torch")

    if filter == "anomaly":
        dataset = dataset.filter(lambda x: x["binary_target"] == 1, num_proc=num_workers)
    elif filter == "normal":
        dataset = dataset.filter(lambda x: x["binary_target"] == 0, num_proc=num_workers)

    return dataset.map(UniformSubsampleOrPad(max_seq_len=max_seq_len), num_proc=num_workers).remove_columns(
        ["id", "multilabel_target", "frame_annotations"]
    )


def test_loader(
    config_name: Literal["i3d_rgb", "swin_rgb"],
    streaming: bool,
    num_workers: Optional[int] = None,
    filter: Literal["anomaly", "normal", "all"] = "all",
):
    if streaming:
        test_ds = streaming_test_dataset(config_name)
    else:
        test_ds = cached_test_dataset(config_name, num_workers)

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=5,  # This is always 5 to match the number of crops
        shuffle=False,  # Must be False to match the number of crops
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=test_batch_collate,
    )

    return test_loader


def train_loader(
    config_name: Literal["i3d_rgb", "swin_rgb"],
    batch_size: int,
    max_seq_len: int,
    streaming: bool,
    num_workers: Optional[int],
    filter: Literal["anomaly", "normal", "all"] = "all",
    shuffle: bool = True,
    seed: Optional[int] = None,
):
    # Shuffling is done in the dataloader for cached dataset, but not for streaming.
    # For streaming dataset, shuffling is done in the dataset itself by shuffling the shards on huffingface hub
    if streaming:
        train_ds = streaming_train_dataset(config_name, max_seq_len, shuffle=shuffle, seed=seed, filter=filter)
        return torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        train_ds = cached_train_dataset(config_name, max_seq_len, num_workers, filter=filter)
        return torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":
    loader = train_loader("i3d_rgb", batch_size=128, max_seq_len=100, streaming=True, num_workers=0, filter="all")
    loader = test_loader("i3d_rgb", streaming=True, num_workers=0)
    # TODO: Inspect the cache_dir bug with custom dataset loading script
    datasets.load_dataset("jherng/xd-violence", "i3d_rgb", split="test", streaming=False, cache_dir=None)
    print(loader.dataset.n_shards)
    # math.ceil(loader.dataset.n_shards * 5 / loader.batch_size)
