import torch
import torch.nn as nn


class HingeLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): The logits. Shape (B, 1). Range [-inf, inf].
            labels (torch.Tensor): The labels. Shape (B, 1). Must be -1 or 1.

        Returns:
            torch.Tensor: The computed loss. Shape (1,).
        """

        assert logits.shape == labels.shape, "Logits and labels must have the same shape."

        loss = torch.mean(torch.max(torch.zeros_like(logits), self.margin - labels * logits))

        return loss
