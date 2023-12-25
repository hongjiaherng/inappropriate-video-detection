from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullHLNet(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.fuse = FusionBlock(feature_dim=feature_dim)
        self.hlc_approx = HLCApproximator()
        self.score_branch = ScoreBranch()

    def forward(self, inputs, seq_len):
        print(f"original inputs: {inputs.shape}")
        # Fused feature
        x = self.fuse(inputs)  # (B, T, D) -> (B, T, 128)
        print(f"fused inputs: {x.shape}")

        # HLC approximator
        hlc_x = self.hlc_approx(x)  # (B, T, 128) -> (B, T, 1)
        print(f"hlc_x: {hlc_x.shape}")

        # Score branch
        self.score_branch(hlc_x.detach(), seq_len) # (B, T, 1) -> (B, T, T)

        # HL-Net

        return hlc_x


class FusionBlock(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1d1 = nn.Conv1d(
            in_channels=feature_dim, out_channels=512, kernel_size=1, padding=0
        )
        self.conv1d2 = nn.Conv1d(
            in_channels=512, out_channels=128, kernel_size=1, padding=0
        )
        self.dropout = nn.Dropout(0.6)

    def forward(self, inputs):
        # Fusion
        x = torch.permute(inputs, [0, 2, 1])  # (B, T, D) -> (B, D, T)
        x = F.relu(self.conv1d1(x))  # (B, D, T) -> (B, 512, T)
        x = self.dropout(x)
        x = F.relu(self.conv1d2(x))  # (B, 512, T) -> (B, 128, T)
        x = self.dropout(x)
        x = torch.permute(x, [0, 2, 1])  # (B, 128, T) -> (B, T, 128)

        return x


class HLCApproximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d1",
                        nn.Conv1d(
                            in_channels=128, out_channels=64, kernel_size=1, padding=0
                        ),
                    ),
                    ("relu1", nn.ReLU()),
                    (
                        "conv1d2",
                        nn.Conv1d(
                            in_channels=64, out_channels=32, kernel_size=1, padding=0
                        ),
                    ),
                    ("relu2", nn.ReLU()),
                ]
            )
        )

        self.classifier = nn.Conv1d(
            in_channels=32, out_channels=1, kernel_size=5, padding=0
        )

    def forward(self, inputs):
        x = torch.permute(inputs, [0, 2, 1])  # (B, T, 128) -> (B, 128, T)
        x = self.backbone(x)  # (B, 128, T) -> (B, 32, T)
        x = F.pad(x, [4, 0], mode="constant", value=0)  # (B, 32, T) -> (B, 32, 4+T)
        x = self.classifier(x)  # (B, 32, 4+T) -> (B, 1, T)
        x = torch.permute(x, [0, 2, 1])  # (B, 1, T) -> (B, T, 1)

        return x


class ScoreBranch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, seq_len=None):
        # logits: (B, T, 1)
        # seq_len: (B,)
        scores = F.sigmoid(logits).squeeze()  # (B, T, 1) -> (B, T)
        pairwise_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # (B, T, T)
        rel_mat = 1.0 - torch.abs(pairwise_diff)
        rel_mat = F.sigmoid((rel_mat - 0.5) / 0.1)  # (B, T, T)

        # Normalize along last dim where:
        # out_mat[0, i, :] represent the closeness of score of clip i to the rest of the clips j = 0,...,T (in prob distribution) for video 0
        out_mat = torch.zeros_like(rel_mat)
        if seq_len is not None:
            for i in range(logits.shape[0]):  # Iterate over batch dim
                rel_mat_i = rel_mat[i, : seq_len[i], : seq_len[i]]
                out_mat[i, : seq_len[i], : seq_len[i]] = F.softmax(rel_mat_i, dim=-1)
        else:
            out_mat = F.softmax(rel_mat, dim=-1)

        return out_mat
