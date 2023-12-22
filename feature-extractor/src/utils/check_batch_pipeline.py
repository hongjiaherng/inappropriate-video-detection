# Script to check the correctness of pipeline of the dataset

import gc
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2 as tv_transforms  # noqa: E402

ROOT_PATH = os.path.abspath(os.path.join(__file__, "../../.."))


def run():
    output_dir = Path(os.path.join(ROOT_PATH, "data", "outputs", "test")).as_posix()
    temp_dir = Path(os.path.join(output_dir, "temp")).as_posix()

    clip_len = 32
    sampling_rate = 2
    batch_size = 4
    hf_dataset_name = "jherng/xd-violence"
    model_name = (
        "i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb"
    )
    num_clips_per_video = -1
    crop_type = "5-crop"  # "5-crop", "10-crop", "center"
    io_backend = "http"
    force_stop_after_n_videos = 1
    check_progress = Path(os.path.join(ROOT_PATH, "results", "progress.txt")).as_posix()

    # Check current progress
    progress = set()  # dataset_utils.get_progress(output_dir, check_progress)

    print(f"{output_dir=}")
    print(f"{temp_dir=}")
    print(f"{batch_size=}")
    print(f"{hf_dataset_name=}")
    print(f"{model_name=}")
    print(f"{num_clips_per_video=}")
    print(f"{crop_type=}")
    print(f"{io_backend=}")
    print(f"{force_stop_after_n_videos=}")
    print(f"{check_progress=}")
    print(f"{len(progress)=}")

    # Initialize dataset
    dataset, _ = dataset_utils.init_hf_dataset(hf_dataset_name, progress)

    video2clips = tv_transforms.Compose(
        [
            custom_transforms.AdaptDataFormat(id_key="id", path_key="path"),
            custom_transforms.VideoReaderInit(io_backend="http"),
            custom_transforms.TemporalClipSample(
                clip_len=clip_len,
                sampling_rate=sampling_rate,
                num_clips=num_clips_per_video,
            ),
            custom_transforms.ClipBatching(batch_size=batch_size),
            custom_transforms.BatchDecodeIter(),
        ]
    )

    crop = custom_transforms.CenterCrop(size=224)
    if crop_type == "5-crop":
        crop = custom_transforms.FiveCrop(size=224)
    elif crop_type == "10-crop":
        crop = custom_transforms.TenCrop(size=224)

    clip2tensor = tv_transforms.Compose(
        [
            custom_transforms.Resize(size=256),
            crop,  # either 5-crop, 10-crop or center
            custom_transforms.ToDType(dtype=torch.float32, scale=True),
            custom_transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            custom_transforms.ConvertTCHWToCTHW(lead_dims=2),
            custom_transforms.PackInputs(preserved_meta=["id", "filename", "batch_id"]),
        ]
    )

    for i, data in enumerate(dataset):
        if i == force_stop_after_n_videos:
            break

        print(f"{i=}")
        print(f"{data=}")

        batch_iter = video2clips(data)

        for j, batch in enumerate(batch_iter):
            print(f"{j=}")
            print(f"{batch['meta']=}")
            print(f"{batch['inputs'].shape=}")
            batch = clip2tensor(batch)

            print(f"{batch['inputs'].shape=}")
            print(f"{batch['meta']=}")
            print()

            for k in range(batch["inputs"].shape[0]):
                plt.figure(figsize=(10, 10))
                plt.suptitle(f"{k}-th clip of {j}-th batch")

                num_frames_to_display = 5
                step_size = max(
                    1, batch["inputs"].shape[3] // (num_frames_to_display - 1)
                )

                for frame_idx in range(num_frames_to_display):
                    actual_frame_idx = min(
                        frame_idx * step_size, batch["inputs"].shape[3] - 1
                    )

                    for crop_idx in range(batch["inputs"].shape[1]):
                        image_array = (
                            batch["inputs"][k, crop_idx, :, actual_frame_idx, :, :]
                            .numpy()
                            .transpose(1, 2, 0)
                        )
                        # Normalize pixel values to [0, 1]
                        image_array = (image_array - np.min(image_array)) / (
                            np.max(image_array) - np.min(image_array)
                        )
                        plt.subplot(
                            batch["inputs"].shape[1],
                            num_frames_to_display,
                            num_frames_to_display * crop_idx + frame_idx + 1,
                        )
                        plt.imshow(image_array)
                        plt.tick_params(
                            axis="both",
                            which="both",
                            bottom=False,
                            top=False,
                            left=False,
                            right=False,
                            labelbottom=False,
                            labelleft=False,
                        )
                        if crop_idx == 0:
                            plt.title(
                                f"{actual_frame_idx}-th frame\n{crop_type}", fontsize=8
                            )

                plt.tight_layout()
                plt.show()

            del batch
            gc.collect()


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
    import dataset_utils
    import transforms as custom_transforms

    run()
