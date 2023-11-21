import io
import urllib
from typing import Dict, Literal

import decord
import numpy as np
import torch


class VideoReaderInit(torch.nn.Module):
    def __init__(self, io_backend: Literal["http", "local"]):
        super().__init__()
        self.io_backend = io_backend

        if self.io_backend == "http":

            def get_buffer(fpath: str) -> bytes:
                return urllib.request.urlopen(fpath).read()

            self.get_buffer = get_buffer

        elif self.io_backend == "local":

            def get_buffer(fpath: str) -> bytes:
                with open(fpath, "rb") as f:
                    return f.read()

            self.get_buffer = get_buffer

        else:
            raise ValueError(
                f'io_backend="{self.io_backend}" is not supported. Please use "http" or "local" instead.'
            )

    def forward(self, results: Dict) -> Dict:
        container = self._get_video_reader(results["filepath"])
        results["total_frames"] = len(container)
        results["video_reader"] = container
        results["avg_fps"] = container.get_avg_fps()
        return results

    def _get_video_reader(self, filepath: str) -> decord.video_reader.VideoReader:
        buf = self.get_buffer(filepath)
        f_obj = io.BytesIO(buf)
        container = decord.VideoReader(f_obj)
        return container

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}(io_backend={self.io_backend})"
        return repr_str


class TemporalClipSample(torch.nn.Module):
    def __init__(self, clip_len: int = 32, sampling_rate: int = 2, num_clips: int = -1):
        super().__init__()
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.sampling_rate = sampling_rate

    def forward(self, results: Dict) -> Dict:
        total_frames = results["total_frames"]

        if self.num_clips == -1:
            max_num_clips = total_frames // (self.clip_len * self.sampling_rate)
        else:
            max_num_clips = min(
                self.num_clips, total_frames // (self.clip_len * self.sampling_rate)
            )

        indices_all = np.arange(
            0, max_num_clips * self.clip_len * self.sampling_rate, self.sampling_rate
        ).reshape((max_num_clips, self.clip_len))

        results["indices_all"] = indices_all  # (num_clips, clip_len)
        results["num_clips"] = max_num_clips
        results["clip_len"] = self.clip_len
        results["sampling_rate"] = self.sampling_rate

        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(clip_len={self.clip_len}, num_clips={self.num_clips}, sampling_rate={self.sampling_rate})"


class VideoDecode(torch.nn.Module):
    def forward(self, results: Dict) -> Dict:
        container = results["video_reader"]
        indices_all = results["indices_all"]
        num_clips = results["num_clips"]
        clip_len = results["clip_len"]

        flat_indices_all = np.ravel(indices_all)
        clips = container.get_batch(flat_indices_all).asnumpy()
        clips = np.reshape(clips, (num_clips, clip_len, *clips.shape[1:]))

        results["clips"] = clips

        return results


# resize, crop, pack

# Resize, FiveCrop,
