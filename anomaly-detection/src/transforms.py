from typing import Any, Dict

import numpy as np
import torch


class ExtractFrameGTs(torch.nn.Module):
    def __init__(self, clip_len: int = 32, sampling_rate: int = 2):
        super().__init__()
        self.clip_len = clip_len
        self.sampling_rate = sampling_rate

    def forward(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create frame-level ground truth labels from annotation metadata for a given video.

        Required keys:
        ```
        * id: str
        * frame_annotations: Dict[str, torch.Tensor]
        * feature: torch.Tensor    # (T, D)
        ```

        Resulting keys:
        ```
        + frame_gts: torch.Tensor  # (T,)
        ```
        """
        num_frames = x["feature"].shape[0] * self.clip_len * self.sampling_rate
        frame_gts = torch.zeros((num_frames,), dtype=torch.bool)
        crop_idx = int(x["id"].split("__")[-1])

        if crop_idx != 0:  # Skip if not the first crop, processing the first crop is enough
            x["frame_gts"] = frame_gts
            return x

        for start, end in zip(x["frame_annotations"]["start"], x["frame_annotations"]["end"]):
            frame_gts[start:end] = 1

        x["frame_gts"] = frame_gts

        return x


class UniformSubsampleOrPad(torch.nn.Module):
    """
    Uniformly subsample or pad a feature tensor to a fixed length.
    (T, D) -> (max_seq_len, D)

    Required keys:
    ```
    * feature: torch.Tensor     # (T, D)
    ```

    Resulting keys:
    ```
    ~ feature: torch.Tensor     # (max_seq_len, D)
    ```

    """

    def __init__(self, max_seq_len: int = 200):
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, x: Dict[str, Any]) -> Dict[str, Any]:
        feature = x["feature"]

        if isinstance(feature, np.ndarray):
            feature = torch.from_numpy(feature)

        # Pad if shorter than max_seq_len otherwise extract uniformly
        if len(feature) > self.max_seq_len:
            feature = self._uniform_subsample(feature)
        else:
            feature = self._pad(feature)

        x["feature"] = feature

        return x

    def _uniform_subsample(self, feat: torch.Tensor):
        # TODO: Implement overlapping temporal subsample
        r = torch.linspace(0, len(feat) - 1, self.max_seq_len, dtype=torch.int32)
        return feat[r, :]

    def _pad(self, feat: torch.Tensor):
        # pad last dim by (0, 0) and 2nd to last by (0, max_seq_len - len(feat))
        return torch.nn.functional.pad(feat, pad=(0, 0, 0, self.max_seq_len - len(feat)), mode="constant", value=0)
