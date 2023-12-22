import os
import sys
import importlib
import datasets

import torch


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

    import transforms as my_transforms

    i3d = importlib.import_module(
        "models.i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb"
    )

    full_pipe = i3d.build_end2end_pipeline(
        io_backend="local",
        id_key="id",
        path_key="path",
        num_clips=-1,
        crop_type="5-crop",
    )

    step_by_step_pipe = [
        my_transforms.AdaptDataFormat(
            id_key="id",
            path_key="path",
        ),
        my_transforms.VideoReaderInit(io_backend="local"),
        my_transforms.TemporalClipSample(
            clip_len=32,
            sampling_rate=2,
            num_clips=-1,
        ),
        my_transforms.VideoDecode(),
        my_transforms.Resize(size=256),
        my_transforms.FiveCrop(size=224),
        my_transforms.ToDType(dtype=torch.float32, scale=True),
        my_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        my_transforms.ConvertTCHWToCTHW(lead_dims=2),
        my_transforms.PackInputs(preserved_meta=["id", "filename"]),
    ]

    train_ds = datasets.load_dataset(
        "jherng/xd-violence", name="video", split="train", streaming=False, num_proc=4
    ).map(
        remove_columns=[
            "binary_target",
            "multilabel_target",
            "frame_annotations",
        ]
    )  # Remove unused columns for preprocessing

    test_ds = datasets.load_dataset(
        "jherng/xd-violence", name="video", split="test", streaming=False, num_proc=4
    ).map(
        remove_columns=[
            "binary_target",
            "multilabel_target",
            "frame_annotations",
        ]
    )

    # Concatenate train and test datasets
    combined_ds = datasets.concatenate_datasets([train_ds, test_ds])

    print(combined_ds)

    for i, ex in enumerate(combined_ds):
        print(f"video: {i}")
        print(f"{ex=}")
        print()

        out1 = full_pipe(ex)

        print(f"{out1['meta']=}")
        print(f"{out1['inputs'].shape=}")
        print(f"{out1['inputs'].mean()=}")
        print(f"{out1['inputs'].std()=}")
        print()

        out2 = ex
        for step in step_by_step_pipe:
            out2 = step(out2)
            print(f"{step=}")
            print(f"{out2['meta']=}")

            if "inputs" in out2:
                print(f"{out2['inputs'].shape=}")
            print()

        print(f"{out2['inputs'].mean()=}")
        print(f"{out2['inputs'].std()=}")

        break
