import io
import gc
import math
import urllib
from typing import Dict, Literal, Sequence, Union, Iterator

import numpy as np
import decord
import torch
import torchvision
import torchvision.transforms.v2.functional as F

torchvision.disable_beta_transforms_warning()


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
    + meta.video_reader: decord.VideoReader
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

    def _get_video_reader(self, filename: str) -> decord.VideoReader:
        buf = self._get_buffer(filename)
        f_obj = io.BytesIO(buf)
        container = decord.VideoReader(f_obj)

        f_obj.close()

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

    def __init__(
        self,
        clip_len: int,
        sampling_rate: int,
        num_clips: int = -1,
        drop_last: bool = False,
        oob_option: Literal["loop", "repeat_last"] = "loop",
    ):
        super().__init__()
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.sampling_rate = sampling_rate
        self.drop_last = drop_last
        self.oob_option = oob_option
        assert self.oob_option in {
            "loop",
            "repeat_last",
        }, f"oob_option={self.oob_option} is not supported. Please use 'loop' or 'repeat_last' instead."

    def forward(self, result: Dict) -> Dict:
        total_frames = result["meta"]["total_frames"]

        if self.num_clips == -1:
            if self.drop_last:
                max_num_clips = total_frames // (self.clip_len * self.sampling_rate)
                frame_indices = np.arange(
                    0,
                    self.clip_len * self.sampling_rate * max_num_clips,
                    self.sampling_rate,
                )

            else:
                max_num_clips = math.ceil(
                    total_frames / (self.clip_len * self.sampling_rate)
                )
                frame_indices = np.arange(
                    0,
                    self.clip_len * self.sampling_rate * max_num_clips,
                    self.sampling_rate,
                )

                if np.any(frame_indices >= total_frames):
                    if self.oob_option == "loop":
                        frame_indices = self._apply_loop_last_clip_op(
                            frame_indices, total_frames
                        )

                    elif self.oob_option == "repeat_last":
                        frame_indices = self._apply_repeat_last_op(
                            frame_indices, total_frames
                        )

        else:
            max_num_clips = self.num_clips
            frame_indices = np.arange(
                0,
                self.clip_len * self.sampling_rate * max_num_clips,
                self.sampling_rate,
            )

            if np.any(frame_indices >= total_frames):
                if self.oob_option == "loop":
                    frame_indices = self._apply_loop_video_op(
                        frame_indices, total_frames
                    )

                elif self.oob_option == "repeat_last":
                    frame_indices = self._apply_repeat_last_op(
                        frame_indices, total_frames
                    )

        result["meta"]["frame_indices"] = frame_indices  # (num_clips, clip_len)
        result["meta"]["num_clips"] = max_num_clips
        result["meta"]["clip_len"] = self.clip_len
        result["meta"]["sampling_rate"] = self.sampling_rate

        return result

    def _apply_loop_last_clip_op(self, frame_indices, total_frames):
        last_clip_start_frame = (
            (total_frames // (self.clip_len * self.sampling_rate))
            * self.clip_len
            * self.sampling_rate
        )
        last_clip_start_idx = np.argmax(frame_indices >= last_clip_start_frame)
        valid_frames = np.arange(
            last_clip_start_frame, total_frames, self.sampling_rate
        )
        last_clip_frames = np.tile(
            valid_frames, math.ceil(self.clip_len / len(valid_frames))
        )[: self.clip_len]
        frame_indices[last_clip_start_idx:] = last_clip_frames

        return frame_indices

    def _apply_loop_video_op(self, frame_indices, total_frames):
        frame_indices = np.mod(frame_indices, total_frames)
        return frame_indices

    def _apply_repeat_last_op(self, frame_indices, total_frames):
        last_valid_frame = np.max(frame_indices[frame_indices < total_frames])
        frame_indices[frame_indices >= total_frames] = last_valid_frame
        return frame_indices

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(clip_len={self.clip_len}, num_clips={self.num_clips}, sampling_rate={self.sampling_rate}, drop_last={self.drop_last}, oob_option={self.oob_option})"


class ClipBatching(torch.nn.Module):
    """
    Batching the frame indices of clips into batches of frame indices of clips.

    Required keys:
    ```
    * meta.frame_indices: np.ndarray # (num_clips * clip_len,)
    ```

    Resulting keys (unspecified keys are preserved):
    ```
    + meta.batch_data: List[{
        "batch_id": int,
        "frame_indices": np.ndarray,  # (batch_size * clip_len,)
        "num_clips": int,
    }] # len (num_batches, )
    - meta.frame_indices: np.ndarray  # (num_clips * clip_len,)
    ```

    """

    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def forward(self, result: Dict):
        frame_indices = result["meta"]["frame_indices"]
        clip_len = result["meta"]["clip_len"]
        num_clips = frame_indices.shape[0] // clip_len

        batch_data_list = []

        for batch_id, batch_start_idx in enumerate(
            range(0, num_clips, self.batch_size)
        ):
            batch_frame_indices = frame_indices[
                batch_start_idx
                * clip_len : (batch_start_idx + self.batch_size)
                * clip_len
            ]  # (batch_size * clip_len, )
            batch_num_clips = batch_frame_indices.shape[0] // clip_len

            batch_data = {
                "batch_id": batch_id,
                "frame_indices": batch_frame_indices,
                "num_clips": batch_num_clips,
            }

            batch_data_list.append(batch_data)

        # Add the batch_data_list to the meta dictionary
        result["meta"]["batch_data"] = batch_data_list

        result["meta"].pop("frame_indices")  # pop the video level frame indices

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(batch_size={self.batch_size})"


class BatchDecodeIter(torch.nn.Module):

    """
    Decodes video clips based on frame indices from a given VideoReader.

    Required keys:
    ```
    * meta.batch_data: List[{
        "batch_id": int,
        "frame_indices": np.ndarray,  # (batch_size * clip_len,)
        "num_clips": int,
    }] # len (num_batches, )
    * meta.video_reader: decord.VideoReader
    * meta.clip_len: int
    ```

    Resulting keys (unspecified keys are preserved):
    ```
    + inputs: torch.Tensor  # (batch_size = num_clips, clip_len, C, H, W)
    + meta.batch_id: int
    + meta.frame_shape: Tuple[int, int]
    + meta.original_frame_shape: Tuple[int, int]
    +/~ meta.num_clips: int
    +/~ meta.total_frames: int
    - meta.batch_data: List[{
        "batch_id": int,
        "frame_indices": np.ndarray,  # (batch_size * clip_len,)
        "num_clips": int,
    }] # len (num_batches, )
    - meta.video_reader: decord.video_reader.VideoReader
    ```
    """

    def __init__(self):
        super().__init__()
        self.batch_data_list = None
        self.vr = None
        self.misc_meta = None
        self.clip_len = None
        self.current_batch_index = 0

    def __len__(self) -> int:
        return len(self.batch_data_list)

    def __iter__(self) -> Iterator[Dict]:
        return self

    def __next__(self) -> Dict:
        if (
            self.batch_data_list is None
            or self.vr is None
            or self.current_batch_index == len(self.batch_data_list)
        ):
            del self.vr, self.batch_data_list, self.misc_meta, self.clip_len
            gc.collect()

            raise StopIteration

        batch_in = self.batch_data_list[self.current_batch_index]
        frame_indices = batch_in["frame_indices"]

        # reset container to the beginning (https://github.com/dmlc/decord/issues/197)
        self.vr.seek(0)
        inputs = self.vr.get_batch(frame_indices).asnumpy()  # (N*T, H, W, C)
        inputs = torch.tensor(inputs, dtype=torch.uint8)
        inputs = torch.permute(inputs, (0, 3, 1, 2))  # (N*T, C, H, W)
        inputs = torch.reshape(
            inputs, (batch_in["num_clips"], self.clip_len, *inputs.shape[1:])
        )  # (N, T, C, H, W)

        batch_out = {
            "inputs": inputs,
            "meta": {
                **self.misc_meta,  # Add the remaining keys from result
                "batch_id": batch_in["batch_id"],
                "num_clips": batch_in["num_clips"],
                "original_frame_shape": tuple(inputs.shape[-2:]),
                "frame_shape": tuple(inputs.shape[-2:]),
                "total_frames": len(frame_indices),  # update total frames in a batch
            },
        }

        self.current_batch_index += 1

        return batch_out

    def forward(self, result: Dict):
        self.batch_data_list = result["meta"]["batch_data"]
        self.clip_len = result["meta"]["clip_len"]
        self.vr = result["meta"]["video_reader"]
        # remaining keys from result
        self.misc_meta = {
            k: v
            for k, v in result["meta"].items()
            if k not in {"batch_data", "video_reader"}
        }
        self.current_batch_index = 0

        return iter(self)


class VideoDecode(torch.nn.Module):
    """
    Decodes video clips based on frame indices from a given VideoReader.

    Required keys:
    ```
    * meta.video_reader: decord.VideoReader
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

        # reset container to the beginning (https://github.com/dmlc/decord/issues/197)
        container.seek(0)
        clips = container.get_batch(frame_indices).asnumpy()  # (N*T, H, W, C)
        clips = torch.tensor(clips, dtype=torch.uint8)
        clips = torch.permute(clips, (0, 3, 1, 2))  # (N*T, C, H, W)
        clips = torch.reshape(
            clips, (num_clips, clip_len, *clips.shape[1:])
        )  # (N, T, C, H, W)

        result["inputs"] = clips
        result["meta"]["original_frame_shape"] = tuple(clips.shape[-2:])
        result["meta"]["frame_shape"] = tuple(clips.shape[-2:])

        # free memory of video_reader, frame_indices
        result["meta"].pop("video_reader")
        result["meta"].pop("frame_indices")
        del frame_indices, container
        gc.collect()

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
        clips = clips.type(self.dtype)
        if self.scale:
            clips /= 255.0
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
        clips = F.normalize(clips, mean=self.mean, std=self.std, inplace=True)

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

    def __init__(self, preserved_meta: Sequence[str] = ["id", "filename", "batch_id"]):
        super().__init__()
        self.preserved_meta = preserved_meta

    def forward(self, result: Dict) -> Dict:
        meta_to_pop = [k for k in result["meta"].keys() if k not in self.preserved_meta]

        if len(meta_to_pop) == len(result["meta"]):
            result.pop("meta")
        else:
            for k in meta_to_pop:
                result["meta"].pop(k)

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(preserved_meta={self.preserved_meta})"
