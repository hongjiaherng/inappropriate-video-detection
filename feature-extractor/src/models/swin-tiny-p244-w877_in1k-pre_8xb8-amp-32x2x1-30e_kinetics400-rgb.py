import os
from typing import Literal

import torch
import torch.nn as nn
import torchvision.transforms.v2 as tv_transforms
import transforms as custom_transforms
from mmaction.registry import MODELS

root_path = os.path.abspath(os.path.join(__file__, "../../.."))  # feature-extractor/

pretrained_path = os.path.join(
    root_path,
    "pretrained",
    "swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-241016b2.pth",
)


preprocessing_cfg = dict(
    io_backend=None,  # to be supplied by upstream
    id_key=None,  # to be supplied by upstream
    path_key=None,  # to be supplied by upstream
    num_clips=None,  # to be supplied by upstream
    crop_type=None,  # to be supplied by upstream
    batch_size=None,  # to be supplied by upstream
    clip_len=32,
    sampling_rate=2,
    resize_size=256,
    crop_size=224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


backbone_cfg = dict(
    type="SwinTransformer3D",
    arch="tiny",
    pretrained=None,
    pretrained2d=False,
    patch_size=(2, 4, 4),
    window_size=(8, 7, 7),
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    patch_norm=True,
)


class ClipEmbeddingExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def extract_feat(self, inputs):
        inputs = self.backbone(inputs)  # (N, C, 4, 7, 7)
        inputs = self.avg_pool(inputs)  # (N, C, 1, 1, 1)
        return inputs.view(inputs.shape[0], -1)  # (N, C)

    def forward(self, inputs):
        return self.extract_feat(inputs)


def build_model() -> nn.Module:
    model = ClipEmbeddingExtractor(backbone=backbone_cfg)
    checkpoint = torch.load(pretrained_path, map_location="cpu")["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def build_video2clips_pipeline(
    batch_size: int,
    io_backend: Literal["http", "local"],
    id_key: str = "id",
    path_key: str = "path",
    num_clips: int = -1,
) -> nn.Module:
    """
    Takes in a whole video and returns a tensor of shape (num_clips, num_crops, num_channels, clip_len, crop_h, crop_w) = (num_clips, num_crops, 3, 32, 224, 224).
    """

    preprocessing_cfg["io_backend"] = io_backend
    preprocessing_cfg["id_key"] = id_key
    preprocessing_cfg["path_key"] = path_key
    preprocessing_cfg["num_clips"] = num_clips
    preprocessing_cfg["batch_size"] = batch_size

    pipeline = [
        custom_transforms.AdaptDataFormat(
            id_key=preprocessing_cfg["id_key"],
            path_key=preprocessing_cfg["path_key"],
        ),
        custom_transforms.VideoReaderInit(io_backend=preprocessing_cfg["io_backend"]),
        custom_transforms.TemporalClipSample(
            clip_len=preprocessing_cfg["clip_len"],
            sampling_rate=preprocessing_cfg["sampling_rate"],
            num_clips=preprocessing_cfg["num_clips"],
        ),
        custom_transforms.ClipsBatching(batch_size=preprocessing_cfg["batch_size"]),
    ]

    return tv_transforms.Compose(pipeline)


def build_clip_pipeline(
    crop_type: Literal["10-crop", "5-crop", "center"] = "5-crop",
) -> nn.Module:
    """
    Takes in a whole video and returns a tensor of shape (num_clips, num_crops, num_channels, clip_len, crop_h, crop_w) = (num_clips, num_crops, 3, 32, 224, 224).
    """

    preprocessing_cfg["crop_type"] = crop_type

    crop_type_config = {
        "5-crop": custom_transforms.FiveCrop,
        "10-crop": custom_transforms.TenCrop,
        "center": custom_transforms.CenterCrop,
    }

    pipeline = [
        custom_transforms.VideoDecode(),
        custom_transforms.Resize(size=preprocessing_cfg["resize_size"]),
        crop_type_config[preprocessing_cfg["crop_type"]](
            size=preprocessing_cfg["crop_size"]
        ),
        custom_transforms.ToDType(dtype=torch.float32, scale=True),
        custom_transforms.Normalize(
            mean=preprocessing_cfg["mean"], std=preprocessing_cfg["std"]
        ),
        custom_transforms.ConvertTCHWToCTHW(lead_dims=2),
        custom_transforms.PackInputs(preserved_meta=["id", "filename", "batch_id"]),
    ]

    return tv_transforms.Compose(pipeline)


def build_end2end_pipeline(
    io_backend: Literal["http", "local"],
    id_key: str = "id",
    path_key: str = "path",
    num_clips: int = -1,
    crop_type: Literal["10-crop", "5-crop", "center"] = "5-crop",
) -> nn.Module:
    """
    Takes in a whole video and returns a tensor of shape (num_clips, num_crops, num_channels, clip_len, crop_h, crop_w) = (num_clips, num_crops, 3, 32, 224, 224).
    """

    preprocessing_cfg["io_backend"] = io_backend
    preprocessing_cfg["id_key"] = id_key
    preprocessing_cfg["path_key"] = path_key
    preprocessing_cfg["num_clips"] = num_clips
    preprocessing_cfg["crop_type"] = crop_type

    crop_type_config = {
        "5-crop": custom_transforms.FiveCrop,
        "10-crop": custom_transforms.TenCrop,
        "center": custom_transforms.CenterCrop,
    }

    pipeline = [
        custom_transforms.AdaptDataFormat(
            id_key=preprocessing_cfg["id_key"],
            path_key=preprocessing_cfg["path_key"],
        ),
        custom_transforms.VideoReaderInit(io_backend=preprocessing_cfg["io_backend"]),
        custom_transforms.TemporalClipSample(
            clip_len=preprocessing_cfg["clip_len"],
            sampling_rate=preprocessing_cfg["sampling_rate"],
            num_clips=preprocessing_cfg["num_clips"],
        ),
        custom_transforms.VideoDecode(),
        custom_transforms.Resize(size=preprocessing_cfg["resize_size"]),
        crop_type_config[preprocessing_cfg["crop_type"]](
            size=preprocessing_cfg["crop_size"]
        ),
        custom_transforms.ToDType(dtype=torch.float32, scale=True),
        custom_transforms.Normalize(
            mean=preprocessing_cfg["mean"], std=preprocessing_cfg["std"]
        ),
        custom_transforms.ConvertTCHWToCTHW(lead_dims=2),
        custom_transforms.PackInputs(preserved_meta=["id", "filename"]),
    ]

    return tv_transforms.Compose(pipeline)
