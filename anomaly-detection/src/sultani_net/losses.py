from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class MILRankingLoss(nn.Module):
    def __init__(self, lambda_smooth: float = 8e-5, lambda_sparsity: float = 8e-5, seq_len: int = 32):
        super().__init__()
        self.seq_len = seq_len
        self.sparsity = SparsityLoss(lambda_sparsity)
        self.smoothness = SmoothingLoss(lambda_smooth)

    def forward(self, scores_normal: torch.Tensor, scores_anomaly: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            scores_normal (torch.Tensor): Scores from the model for normal instances. Shape (B * T, 1).
            scores_anomaly (torch.Tensor): Scores from the model for anomaly instances. Shape (B * T, 1).

        Returns:
            torch.Tensor: The computed loss.
        """
        batch_size = int(scores_normal.shape[0] / self.seq_len)

        hinge = torch.tensor(0.0, device=scores_normal.device)
        smoothness = torch.tensor(0.0, device=scores_normal.device)
        sparsity = torch.tensor(0.0, device=scores_normal.device)

        for i in range(batch_size):
            max_normal = torch.max(scores_normal[i * self.seq_len : (i + 1) * self.seq_len])  # (T, 1) -> (1, 1)
            max_anonomaly = torch.max(scores_anomaly[i * self.seq_len : (i + 1) * self.seq_len])  # (T, 1) -> (1, 1)

            hinge += F.relu(1 - max_anonomaly + max_normal)
            sparsity += self.sparsity(scores_anomaly[i * self.seq_len : (i + 1) * self.seq_len])
            smoothness += self.smoothness(scores_anomaly[i * self.seq_len : (i + 1) * self.seq_len])

        loss = (hinge + sparsity + smoothness) / batch_size

        return loss, sparsity.detach(), smoothness.detach()


class SmoothingLoss(nn.Module):
    def __init__(self, lambda_: float = 8e-5):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, scores_anomaly: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores_anomaly (torch.Tensor): Scores from the model for anomaly instances. Shape (T, 1).

        Returns:
            torch.Tensor: The computed smoothing term.
        """
        scores_diff = (scores_anomaly[:-1] - scores_anomaly[1:]) ** 2  # (T - 1, 1)
        smoothness = torch.sum(scores_diff)  # (1,)

        return self.lambda_ * smoothness


class SparsityLoss(nn.Module):
    def __init__(self, lambda_: float = 8e-5):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, scores_anomaly: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores_anomaly (torch.Tensor): Scores from the model for anomaly instances. Shape (T, 1).

        Returns:
            torch.Tensor: The computed sparsity term.
        """
        sparsity = torch.sum(scores_anomaly)

        return self.lambda_ * sparsity
