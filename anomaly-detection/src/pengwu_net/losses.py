from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class PengWuNetLoss(nn.Module):
    def __init__(self, lambda_: float = 5.0, is_topk: bool = True, q: int = 16):
        super().__init__()
        self.lambda_ = lambda_
        self.distill_criterion = DistillLoss()
        self.mil_criterion = MILLoss(is_topk=is_topk, q=q)

    def forward(
        self, logits_hl: torch.Tensor, logits_hlc: torch.Tensor, bag_labels: torch.Tensor, seq_len: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        distill_loss = self.distill_criterion(logits_hl, logits_hlc, seq_len)
        mil_loss_hlc = self.mil_criterion(logits_hlc, bag_labels, seq_len)
        mil_loss_hl = self.mil_criterion(logits_hl, bag_labels, seq_len)

        total_loss = self.lambda_ * distill_loss + mil_loss_hlc + mil_loss_hl

        return (
            total_loss,  # Backpropagatable
            distill_loss.detach(),  # Not backpropagatable (just for logging purposes)
            mil_loss_hl.detach(),  # Not backpropagatable
            mil_loss_hlc.detach(),  # Not backpropagatable
        )  # TODO: Might cause memory issue


class DistillLoss(nn.Module):
    """
    Loss to encourage the student model (HLC approximator) to mimic the teacher model's predictions (HL-Net).

    Note:
        This loss function utilizes knowledge distillation by comparing the sigmoid activations
        of the teacher (logits_hl) and student (logits_hlc) logits for each instance in the batch.
        The sequence lengths (seq_len) are used to handle variable-length sequences.
    """

    def forward(self, logits_hl: torch.Tensor, logits_hlc: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits_hl (torch.Tensor): Logits from the teacher model (HL-Net). Shape (B, T, 1).
            logits_hlc (torch.Tensor): Logits from the student model (HLC approximator). Shape (B, T, 1).
            seq_len (torch.Tensor): Sequence lengths for each instance in the batch. Shape (B, 1).

        Returns:
            torch.Tensor: The computed knowledge distillation loss.
        """
        batch_size = logits_hl.shape[0]
        loss = torch.tensor(0.0, device=logits_hl.device)

        if seq_len is not None:
            for i in range(batch_size):
                scores_hl_i = F.sigmoid(
                    logits_hl[i, : seq_len[i], :].squeeze()
                ).detach()  # detach to prevent gradients from being backpropagated through the logits of the teacher model (logits_hl)
                scores_hlc_i = F.sigmoid(logits_hlc[i, : seq_len[i], :].squeeze())
                loss += -torch.sum(scores_hl_i * torch.log(scores_hlc_i))
        else:
            # Vectorized implementation
            scores_hl = F.sigmoid(logits_hl).detach()
            scores_hlc = F.sigmoid(logits_hlc)
            loss = -torch.sum(scores_hl * torch.log(scores_hlc))

        loss /= batch_size

        return loss


class MILLoss(nn.Module):
    """
    Multiple Instance Learning (MIL) loss function for bag-level classification.

    Args:
        is_topk (bool): If True, apply top-k pooling to consider the highest-scoring instances in each bag.
        q (int): Parameter used to calculate top-k value based on sequence length. Default: 16.

    Note:
        This MIL loss function is designed for bag-level classification tasks. It allows for top-k pooling,
        where the highest-scoring instances in each bag are considered for computing the loss.

    """

    def __init__(self, is_topk: bool, q: int = 16) -> None:
        super().__init__()
        self.is_topk = is_topk
        self.q = q
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, logits: torch.Tensor, bag_labels: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Logits for each instance in each bag. Shape (B, T, 1).
            bag_labels (torch.Tensor): Binary labels indicating positive (1) or negative (0) bags. Shape (B, 1).
            seq_len (torch.Tensor): Sequence lengths for each bag in the batch. Shape (B, 1).

        Returns:
            torch.Tensor: The computed MIL loss.
        """
        bag_labels = bag_labels.type(torch.float32)

        if self.is_topk:
            return self._forward_meanpool_topk(logits, bag_labels, seq_len)

        return self._forward_meanpool(logits, bag_labels, seq_len)

    def _k(self, seq_len: torch.Tensor) -> torch.Tensor:
        """
        Calculate top-k value based on sequence length for all bags in the batch.

        Equation:
            k = floor((T/q) + 1)
        """
        return torch.clamp(seq_len // self.q, min=1)

    def _forward_meanpool_topk(self, logits: torch.Tensor, bag_labels: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Logits for each instance in each bag. Shape (B, T, 1).
            bag_labels (torch.Tensor): Binary labels indicating positive (1) or negative (0) bags. Shape (B, 1).
            seq_len (torch.Tensor): Sequence lengths for each bag in the batch. Shape (B, 1).

        Returns:
            torch.Tensor: The computed MIL loss.
        """
        if seq_len is not None:
            bag_logits = torch.zeros(logits.shape[0], device=logits.device)
            ks = self._k(seq_len=seq_len)  # (B, 1)
            for i in range(logits.shape[0]):
                logits_i = logits[i, : seq_len[i], :].squeeze()
                logits_i_topk, _ = torch.topk(logits_i, ks[i], largest=True, sorted=False)
                bag_logits[i] = torch.mean(logits_i_topk)
        else:
            bag_logits = torch.zeros(logits.shape[0], device=logits.device)
            k = self._k(seq_len=torch.tensor(logits.shape[1]))  # (B, 1)
            for i in range(logits.shape[0]):
                logits_i = logits[i, :, :].squeeze()
                logits_i_topk, _ = torch.topk(logits_i, k, largest=True, sorted=False)
                bag_logits[i] = torch.mean(logits_i_topk)

        loss = self.criterion(bag_logits, bag_labels)
        return loss

    def _forward_meanpool(self, logits: torch.Tensor, bag_labels: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Logits for each instance in each bag. Shape (B, T, 1).
            bag_labels (torch.Tensor): Binary labels indicating positive (1) or negative (0) bags. Shape (B, 1).
            seq_len (torch.Tensor): Sequence lengths for each bag in the batch. Shape (B, 1).

        Returns:
            torch.Tensor: The computed MIL loss.
        """
        if seq_len is not None:
            bag_logits = torch.zeros(logits.shape[0], device=logits.device)
            for i in range(logits.shape[0]):
                logits_i = logits[i, : seq_len[i], :].squeeze()
                bag_logits[i] = torch.mean(logits_i)
        else:
            bag_logits = torch.mean(logits, dim=1).squeeze()

        loss = self.criterion(bag_logits, bag_labels)
        return loss

