import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineNet(nn.Module):
    def __init__(self, feature_dim: int, dropout_prob: float = 0.6):
        super().__init__()

        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout_prob)

        self.apply(self._init_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Input features. Shape (B, D).

        Returns:
            torch.Tensor: The computed logits. Shape (B, 1).
        """

        x = self.dropout(F.relu(self.fc1(inputs)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Input features. Shape (B, D).

        Returns:
            torch.Tensor: The predicted labels. Shape (B, 1).
        """

        logits = self.forward(inputs)
        scores = F.sigmoid(logits)

        return scores

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)
