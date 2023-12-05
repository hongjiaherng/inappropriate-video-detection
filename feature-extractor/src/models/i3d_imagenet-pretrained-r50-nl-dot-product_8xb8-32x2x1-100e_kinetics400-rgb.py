# Adapted from:
# - https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/i3d/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb.py
# - https://github.com/open-mmlab/mmaction2/blob/main/configs/_base_/models/i3d_r50.py
# - https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet/blob/main/resnet.py
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
    "i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb_20220812-8e1f2148.pth",
)


backbone_cfg = dict(
    type="ResNet3d",  # https://mmaction2.readthedocs.io/en/latest/api.html#mmaction.models.backbones.ResNet3d
    pretrained2d=False,
    pretrained=None,
    depth=50,
    conv1_kernel=(5, 7, 7),
    conv1_stride_t=2,
    pool1_stride_t=2,
    conv_cfg=dict(type="Conv3d"),
    norm_eval=False,
    inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
    zero_init_residual=False,
    non_local=((0, 0, 0), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), (0, 0, 0)),
    non_local_cfg=dict(
        sub_sample=True,
        use_scale=False,
        norm_cfg=dict(type="BN3d", requires_grad=True),
        mode="dot_product",
    ),
)

preprocessing_cfg = dict(
    io_backend=None,  # to be supplied by upstream
    id_key=None,  # to be supplied by upstream
    path_key=None,  # to be supplied by upstream
    num_clips=None,  # to be supplied by upstream
    crop_type=None,  # to be supplied by upstream
    clip_len=32,
    sampling_rate=2,
    resize_size=256,
    crop_size=224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
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


def build_preprocessing_pipeline(
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
