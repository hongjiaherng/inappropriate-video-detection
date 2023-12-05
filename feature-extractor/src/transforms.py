import io
import urllib
from typing import Dict, Literal, Sequence, Union

import decord
import numpy as np
import torch
import torchvision.transforms.v2.functional as F


class AdaptDataFormat(torch.nn.Module):
    """
    Adapt data format from HuggingFace's Dataset to our format.

    Required keys:
    ```
    * id: str
    * path: str
    ```

    Resulting keys (all keys are overwritten):
    ```
    + meta.id: str
    + meta.filename: str
    - id: str
    - path: str
    ```

    Example:
    --------
    Source data format (HuggingFace's Dataset):
    ```json
    {
        "id": "A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A",
        "path": "https://huggingface.co/datasets/jherng/xd-violence/resolve/main/data/video/1-1004/A.Beautiful.Mind.2001__%2300-01-45_00-02-50_label_A.mp4",
        "binary_target": 0,
        "multilabel_targets": [0],
        "frame_annotations": {"start": [], "end": []}
    }
    ```

    Target data format:
    ```json
    {
        "meta": {
            "id": "A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A",
            "filename": "https://huggingface.co/datasets/jherng/xd-violence/resolve/main/data/video/1-1004/A.Beautiful.Mind.2001__%2300-01-45_00-02-50_label_A.mp4",
        }
    }
    ```
    """

    def __init__(self, id_key: str = "id", path_key: str = "path"):
        super().__init__()
        self.id_key = id_key
        self.path_key = path_key

    def forward(self, result: Dict) -> Dict:
        result = {
            "meta": {
                "id": result[self.id_key],
                "filename": result[self.path_key],
            }
        }

        return result

    def __repr__(self) -> str:
        repr_str = (
            f"{self.__class__.__name__}(id_key={self.id_key}, path_key={self.path_key})"
        )
        return repr_str


class VideoReaderInit(torch.nn.Module):
    """
    Initializes a VideoReader based on the given file path.

    Required keys:
    ```
    * meta.filename: str
    ```

    Resulting keys (unspecified keys are preserved):
    ```
    + meta.total_frames: int
    + meta.video_reader: decord.video_reader.VideoReader
    + meta.avg_fps: float
    ```

    """

    def __init__(self, io_backend: Literal["http", "local"]):
        super().__init__()
        self.io_backend = io_backend

    def forward(self, result: Dict) -> Dict:
        container = self._get_video_reader(result["meta"]["filename"])
        result["meta"]["total_frames"] = len(container)
        result["meta"]["video_reader"] = container
        result["meta"]["avg_fps"] = container.get_avg_fps()

        return result

    def _get_video_reader(self, filename: str) -> decord.video_reader.VideoReader:
        buf = self._get_buffer(filename)
        f_obj = io.BytesIO(buf)
        container = decord.VideoReader(f_obj)
        return container

    def _get_buffer(self, fpath: str) -> bytes:
        if self.io_backend == "http":
            return urllib.request.urlopen(fpath).read()
        elif self.io_backend == "local":
            with open(fpath, "rb") as f:
                return f.read()
        else:
            raise ValueError(
                f'io_backend="{self.io_backend}" is not supported. Please use "http" or "local" instead.'
            )

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}(io_backend={self.io_backend})"
        return repr_str


class TemporalClipSample(torch.nn.Module):
    """
    Samples the frame indices of a number of clips based on clip length and sampling rate.

    Required keys:
    ```
    * meta.total_frames: int
    ```

    Resulting keys (unspecified keys are preserved):
    ```
    + meta.frame_indices: np.ndarray  # (num_clips * clip_len,)
    + meta.num_clips: int
    + meta.clip_len: int
    + meta.sampling_rate: int
    ```

    """

    def __init__(self, clip_len: int, sampling_rate: int, num_clips: int = -1):
        super().__init__()
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.sampling_rate = sampling_rate

    def forward(self, result: Dict) -> Dict:
        total_frames = result["meta"]["total_frames"]

        if self.num_clips == -1:
            max_num_clips = total_frames // (self.clip_len * self.sampling_rate)
        else:
            max_num_clips = min(
                self.num_clips, total_frames // (self.clip_len * self.sampling_rate)
            )

        frame_indices = np.arange(
            0, max_num_clips * self.clip_len * self.sampling_rate, self.sampling_rate
        )
        # len of (num_clips * clip_len)

        result["meta"]["frame_indices"] = frame_indices  # (num_clips * clip_len,)
        result["meta"]["num_clips"] = max_num_clips
        result["meta"]["clip_len"] = self.clip_len
        result["meta"]["sampling_rate"] = self.sampling_rate

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(clip_len={self.clip_len}, num_clips={self.num_clips}, sampling_rate={self.sampling_rate})"


class VideoDecode(torch.nn.Module):
    """
    Decodes video clips based on frame indices from a given VideoReader.

    Required keys:
    ```
    * meta.video_reader: decord.video_reader.VideoReader
    * meta.frame_indices: np.ndarray  # (num_clips * clip_len,)
    * meta.num_clips: int
    * meta.clip_len: int
    ```

    Resulting keys (unspecified keys are preserved):
    ```
    + inputs: torch.Tensor  # (num_clips, clip_len, C, H, W)
    + meta.original_frame_shape: Tuple[int, int]
    + meta.frame_shape: Tuple[int, int]
    - meta.video_reader: decord.video_reader.VideoReader
    - meta.frame_indices: np.ndarray  # (num_clips * clip_len,)
    ```
    """

    def forward(self, result: Dict) -> Dict:
        container = result["meta"]["video_reader"]
        frame_indices = result["meta"]["frame_indices"]
        num_clips = result["meta"]["num_clips"]
        clip_len = result["meta"]["clip_len"]

        clips = container.get_batch(frame_indices).asnumpy()  # (N*T, H, W, C)
        clips = torch.tensor(clips, dtype=torch.uint8)
        clips = torch.permute(clips, (0, 3, 1, 2))  # (N*T, C, H, W)
        clips = torch.reshape(
            clips, (num_clips, clip_len, *clips.shape[1:])
        )  # (N, T, C, H, W)

        # free memory of video_reader, frame_indices
        result["meta"].pop("video_reader")
        del container
        result["meta"].pop("frame_indices")
        del frame_indices

        result["inputs"] = clips
        result["meta"]["original_frame_shape"] = tuple(clips.shape[-2:])
        result["meta"]["frame_shape"] = tuple(clips.shape[-2:])

        return result


class Resize(torch.nn.Module):
    """
    Resize the input video clips.

    Required keys:
    ```
    * inputs: torch.Tensor  # (..., C, H, W)
    ```

    Resulting keys (unspecified keys are preserved):
    ```
    ~ inputs: torch.Tensor  # (..., C, H', W')
    +/~ meta.frame_shape: Tuple[int, int]
    ```
    """

    def __init__(self, size: Union[int, Sequence[int]]):
        super().__init__()
        self.size = size

    def forward(self, result: Dict) -> Dict:
        clips = result["inputs"]
        clips = F.resize(clips, size=self.size, antialias=True)

        result["inputs"] = clips
        result["meta"]["frame_shape"] = tuple(
            clips.shape[-2:]
        )  # last 2 dims surely represent (H, W)

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class FiveCrop(torch.nn.Module):
    """
    Applies the five-crop transformation to the input video clips.

    Required keys:
    ```
    * inputs: torch.Tensor  # (..., T, C, H, W)
    ```

    Resulting keys (unspecified keys are preserved):
    ```
    ~ inputs: torch.Tensor  # (..., 5, T, C, H', W')
    +/~ meta.frame_shape: Tuple[int, int]
    + meta.num_crops: int  # 5
    ```
    """

    def __init__(self, size: Union[int, Sequence[int]]):
        super().__init__()
        self.size = size

    def forward(self, result: Dict) -> Dict:
        clips = result["inputs"]
        clips = F.five_crop(clips, size=self.size)
        clips = torch.stack(clips, dim=-5)  # (..., 5, C, T, H, W)

        result["inputs"] = clips
        result["meta"]["frame_shape"] = tuple(
            clips.shape[-2:]
        )  # last 2 dims surely represent (H, W)
        result["meta"]["num_crops"] = 5

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class TenCrop(torch.nn.Module):
    """
    Applies the ten-crop transformation to the input video clips.

    Required keys:
    ```
    * inputs: torch.Tensor  # (..., T, C, H, W)
    ```

    Resulting keys (unspecified keys are preserved):
    ```
    ~ inputs: torch.Tensor  # (..., 10, T, C, H', W')
    +/~ meta.frame_shape: Tuple[int, int]
    + meta.num_crops: int  # 10
    ```
    """

    def __init__(self, size: Union[int, Sequence[int]]):
        super().__init__()
        self.size = size

    def forward(self, result: Dict) -> Dict:
        clips = result["inputs"]
        clips = F.ten_crop(clips, size=self.size)
        clips = torch.stack(clips, dim=-5)  # (..., 10, C, T, H, W)

        result["inputs"] = clips
        result["meta"]["frame_shape"] = tuple(
            clips.shape[-2:]
        )  # last 2 dims surely represent (H, W)
        result["meta"]["num_crops"] = 10

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class CenterCrop(torch.nn.Module):
    """
    Applies the center-crop transformation to the input video clips.

    Required keys:
    ```
    * inputs: torch.Tensor  # (..., T, C, H, W)
    ```

    Resulting keys (unspecified keys are preserved):
    ```
    ~ inputs: torch.Tensor  # (..., 1, T, C, H', W')
    +/~ meta.frame_shape: Tuple[int, int]
    + meta.num_crops: int  # 1
    ```
    """

    def __init__(self, size: Union[int, Sequence[int]]):
        super().__init__()
        self.size = size

    def forward(self, result: Dict) -> Dict:
        clips = result["inputs"]
        clips = F.center_crop(clips, output_size=self.size)
        clips = torch.unsqueeze(clips, dim=-5)  # (..., 1, C, T, H, W)

        result["inputs"] = clips
        result["meta"]["frame_shape"] = tuple(
            clips.shape[-2:]
        )  # last 2 dims surely represent (H, W)
        result["meta"]["num_crops"] = 1

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class ToDType(torch.nn.Module):
    """
    Converts the data type of the input video clips and scale between 0 and 1.

    Required keys:
    ```
    * inputs: torch.Tensor
    ```

    Resulting keys (unspecified keys are preserved):
    ```
    ~ inputs: torch.Tensor
    ```
    """

    def __init__(self, dtype: torch.dtype = torch.float32, scale: bool = True):
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def forward(self, result: Dict) -> Dict:
        clips = result["inputs"]
        clips = F.to_dtype(clips, dtype=self.dtype, scale=self.scale)

        result["inputs"] = clips

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dtype={self.dtype}, scale={self.scale})"


class Normalize(torch.nn.Module):
    """
    Normalizes the input video clips based on the given mean and std.

    Required keys:
    ```
    * inputs: torch.Tensor  # (..., C, H, W)
    ```

    Resulting keys (unspecified keys are preserved):
    ```
    ~ inputs: torch.Tensor  # (..., C, H, W)
    ```
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, result: Dict) -> Dict:
        clips = result["inputs"]
        clips = F.normalize(clips, mean=self.mean, std=self.std)

        result["inputs"] = clips

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class ConvertTCHWToCTHW(torch.nn.Module):
    """
    Converts the input video clips from TCHW to CTHW format.

    Required keys:
    ```
    * inputs: torch.Tensor  # (..., T, C, H, W)
    ```

    Resulting keys (unspecified keys are preserved):
    ```
    ~ inputs: torch.Tensor  # (..., C, T, H, W)
    ```
    """

    def __init__(self, lead_dims: int):
        super().__init__()
        self.lead_dims = lead_dims
        self.permute_idx = tuple(range(0, self.lead_dims)) + tuple(
            i + self.lead_dims for i in (1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
        )

    def forward(self, result: Dict) -> Dict:
        clips = result["inputs"]
        clips = torch.permute(clips, self.permute_idx)

        result["inputs"] = clips

        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(lead_dims={self.lead_dims})"


class PackInputs(torch.nn.Module):
    """
    Packs the input video clips by preserving specified metadata keys.

    Required keys:
    ```
    * meta: Dict
    ```

    Resulting keys:
    ```
    ~ meta: Dict  # Only specified keys are preserved
    ```
    """

    def __init__(self, preserved_meta: Sequence[str] = ["id", "filename"]):
        super().__init__()
        self.preserved_meta = preserved_meta

    def forward(self, result: Dict) -> Dict:
        meta_to_pop = [k for k in result["meta"].keys() if k not in self.preserved_meta]

        if len(meta_to_pop) == len(result["meta"]):
            result.pop("meta")

        for k in meta_to_pop:
            result["meta"].pop(k)

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(preserved_meta={self.preserved_meta})"
