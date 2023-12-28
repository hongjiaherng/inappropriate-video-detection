# TODO: Half way through checking the implementation of 3 branches, i.e., Holistic, Localized, and Score branches

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullHLNet(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        dropout_prob: float = 0.6,
        hlc_seq_len: int = 5,
        threshold: float = 0.7,
        sigma: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.dropout_prob = dropout_prob
        self.hlc_seq_len = hlc_seq_len
        self.threshold = threshold
        self.sigma = sigma
        self.gamma = gamma

        self.fuse = FusionBlock(
            feature_dim=self.feature_dim, dropout_prob=self.dropout_prob
        )
        self.hlc_approx = HLCApproximator(seq_len=self.hlc_seq_len)
        self.holistic_branch = HolisticBranch(
            threshold=self.threshold, dropout_prob=self.dropout_prob
        )
        self.localized_branch = LocalizedBranch(
            sigma=self.sigma, gamma=self.gamma, dropout_prob=self.dropout_prob
        )
        self.score_branch = ScoreBranch(self.dropout_prob)

    def forward(self, inputs: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        print(f"original inputs: {inputs.shape}")
        # Fused feature
        x_fused = self.fuse(inputs)  # (B, T, D) -> (B, T, 128)
        print(f"fused inputs: {x_fused.shape}")

        # HLC approximator
        hlc_x = self.hlc_approx(x_fused)  # (B, T, 128) -> (B, T, 1)
        print(f"hlc_x: {hlc_x.shape}")

        # Score branch
        score_x = self.score_branch(x_fused, hlc_x.detach(), seq_len)
        print(f"score_x: {score_x.shape}")

        # Holistic branch
        holistic_x = self.holistic_branch(x_fused, inputs, seq_len)
        print(f"holistic_x: {holistic_x.shape}")

        # Localized branch
        localized_x = self.localized_branch(x_fused)
        print(f"localized_x: {localized_x.shape}")

        return hlc_x


class FusionBlock(nn.Module):
    def __init__(self, feature_dim: int, dropout_prob: float = 0.6) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.dropout_prob = dropout_prob

        self.conv1 = nn.Conv1d(self.feature_dim, 512, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(512, 128, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Fusion
        x = torch.permute(inputs, [0, 2, 1])  # (B, T, D) -> (B, D, T)
        x = F.relu(self.conv1(x))  # (B, D, T) -> (B, 512, T)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))  # (B, 512, T) -> (B, 128, T)
        x = self.dropout(x)
        x = torch.permute(x, [0, 2, 1])  # (B, 128, T) -> (B, T, 128)

        return x


class HLCApproximator(nn.Module):
    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len  # sequence length to aggregate over
        self.backbone = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=1, padding=0),
            nn.ReLU(),
        )

        self.classifier = nn.Conv1d(32, 1, kernel_size=seq_len, padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = torch.permute(inputs, [0, 2, 1])  # (B, T, 128) -> (B, 128, T)
        x = self.backbone(x)  # (B, 128, T) -> (B, 32, T)
        x = F.pad(
            x, [self.seq_len - 1, 0], mode="constant", value=0
        )  # (B, 32, T) -> (B, 32, seq_len - 1 + T)
        x = self.classifier(x)  # (B, 32, seq_len - 1 + T) -> (B, 1, T)
        x = torch.permute(x, [0, 2, 1])  # (B, 1, T) -> (B, T, 1)

        return x


class ScoreBranch(nn.Module):
    """Score branch of HL-Net that captures the closeness of scores between clips"""

    def __init__(self, dropout_prob: float) -> None:
        super().__init__()
        self.dropout_prob = dropout_prob

        self.graph_conv1 = GraphConvolution(128, 32, skip_connection=True, bias=True)
        self.graph_conv2 = GraphConvolution(32, 32, skip_connection=True, bias=True)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(
        self,
        x_fused: torch.Tensor,
        hlc_logits: torch.Tensor,
        seq_len: torch.Tensor = None,
    ) -> torch.Tensor:
        # logits: (B, T, 1)
        # seq_len: (B,)

        # Compute adjacency matrix (score closeness prior)
        adj_mat = self._compute_adj_mat(hlc_logits, seq_len)

        x = F.relu(self.graph_conv1(x_fused, adj_mat))
        x = self.dropout(x)
        x = F.relu(self.graph_conv2(x, adj_mat))
        x = self.dropout(x)

        return x

    def _compute_adj_mat(
        self, hlc_logits: torch.Tensor, seq_len: torch.Tensor
    ) -> torch.Tensor:
        # logits: (B, T, 1)
        # seq_len: (B,)
        scores = F.sigmoid(hlc_logits).squeeze()  # (B, T, 1) -> (B, T)
        pairwise_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # (B, T, T)
        temp_mat = 1.0 - torch.abs(pairwise_diff)
        temp_mat = F.sigmoid((temp_mat - 0.5) / 0.1)  # (B, T, T)

        # Normalize along last dim where:
        # adj_mat[0, i, :] represent the closeness of score of clip i to the rest of the clips j = 0,...,T (in prob distribution) for video 0
        adj_mat = torch.zeros_like(temp_mat)
        if seq_len is not None:
            for i in range(hlc_logits.shape[0]):  # Iterate over batch dim
                temp_mat_i = temp_mat[i, : seq_len[i], : seq_len[i]]
                adj_mat[i, : seq_len[i], : seq_len[i]] = F.softmax(temp_mat_i, dim=-1)
        else:
            adj_mat = F.softmax(adj_mat, dim=-1)

        return adj_mat


class HolisticBranch(nn.Module):
    """Holistic branch of HL-Net that captures feature similarity between clips"""

    def __init__(
        self, threshold: float, dropout_prob: float, eps: float = 1e-8
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.dropout_prob = dropout_prob
        self.eps = eps  # epsilon to avoid division by zero

        self.graph_conv1 = GraphConvolution(128, 32, skip_connection=True, bias=True)
        self.graph_conv2 = GraphConvolution(32, 32, skip_connection=True, bias=True)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(
        self, x_fused: torch.Tensor, x_orig: torch.Tensor, seq_len: torch.Tensor = None
    ) -> torch.Tensor:
        # inputs: (B, T, 128)
        # seq_len: (B,)

        # Compute adjacency matrix (similarity prior)
        adj_mat = self._compute_adj_mat(x_orig, seq_len)

        x = F.relu(self.graph_conv1(x_fused, adj_mat))
        x = self.dropout(x)
        x = F.relu(self.graph_conv2(x, adj_mat))
        x = self.dropout(x)

        return x

    def _compute_adj_mat(
        self, x_orig: torch.Tensor, seq_len: torch.Tensor
    ) -> torch.Tensor:
        # inputs: (B, T, D)
        # return: (B, T, T)
        # dot product between pairs of clips (B, T, T)
        pairwise_dot = torch.matmul(x_orig, x_orig.permute(0, 2, 1))

        # norm of each clip for whole batch (B, T, 1)
        clip_norms = torch.norm(x_orig, p=2, dim=2, keepdim=True)

        # multiplication of norms across all clips (B, T, T)
        pairwise_norms = torch.matmul(clip_norms, clip_norms.permute(0, 2, 1))

        # cosine similarity between clips (B, T, T)
        temp_mat = pairwise_dot / (pairwise_norms + self.eps)

        adj_mat = torch.zeros_like(temp_mat)
        if seq_len is not None:
            for i in range(x_orig.shape[0]):  # Iterate over batch dim
                temp_mat_i = temp_mat[i, : seq_len[i], : seq_len[i]]
                temp_mat_i = F.threshold(
                    temp_mat_i, threshold=self.threshold, value=0.0
                )
                adj_mat[i, : seq_len[i], : seq_len[i]] = F.softmax(temp_mat_i, dim=-1)

        else:
            temp_mat = F.threshold(temp_mat, threshold=self.threshold, value=0.0)
            adj_mat = F.softmax(temp_mat, dim=-1)

        return adj_mat


class LocalizedBranch(nn.Module):
    """Localized branch of HL-Net that captures positional distance (proximity) between clips"""

    def __init__(self, sigma: float, gamma: float, dropout_prob: float) -> None:
        super().__init__()
        self.sigma = sigma
        self.gamma = gamma
        self.dropout_prob = dropout_prob

        self.graph_conv1 = GraphConvolution(128, 32, skip_connection=True, bias=True)
        self.graph_conv2 = GraphConvolution(32, 32, skip_connection=True, bias=True)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, x_fused: torch.Tensor) -> torch.Tensor:
        # inputs: (B, T, D)

        # Compute adjacency matrix (proximity prior)
        adj_mat = self._compute_adj_mat(x_fused)

        x = F.relu(self.graph_conv1(x_fused, adj_mat))
        x = self.dropout(x)
        x = F.relu(self.graph_conv2(x, adj_mat))
        x = self.dropout(x)

        return x

    def _compute_adj_mat(self, x_fused: torch.Tensor) -> torch.Tensor:
        # inputs: (B, T, D)
        # return: (B, T, T)
        batch_size, max_seq_len, _ = x_fused.shape

        dist = torch.arange(max_seq_len, dtype=torch.float32).to(x_fused.device)
        adj_mat = torch.abs(dist.unsqueeze(0) - dist.unsqueeze(1))
        adj_mat = torch.exp(-torch.pow(adj_mat, self.gamma) / self.sigma)
        adj_mat = adj_mat.unsqueeze(0).repeat(batch_size, 1, 1)

        return adj_mat


class GraphConvolution(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        skip_connection: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.skip_connection = skip_connection

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        if not self.skip_connection:
            self.proj = lambda x: 0  # no skip connection: (B, T, D_out) + 0
        elif in_features == out_features:
            # skip connection (in_features == out_features): (B, T, D_out=D_in) + (B, T, D_out=D_in)
            self.proj = lambda x: x
        else:
            # skip connection (in_features != out_features): (B, T, D_out) + Linear(B, T, D_in) => (B, T, D_out) + (B, T, D_out) => (B, T, D_out)
            self.proj = nn.Conv1d(in_features, out_features, kernel_size=5, padding=2)

    def reset_parameters(self) -> None:
        # Init weight using Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:  # Init bias to 0.1
            nn.init.constant_(self.bias, 0.1)

    def forward(self, inputs: torch.Tensor, adj_mat: torch.Tensor) -> torch.Tensor:
        # inputs: (B, T, D_in)
        # adj_mat: (B, T, T)
        # return: (B, T, D_out)
        x = torch.matmul(inputs, self.weight.T)  # (B, T, D_out)
        x = torch.matmul(adj_mat, x)  # (B, T, D_out)

        if self.bias is not None:
            x = x + self.bias

        if self.proj is not None and self.in_features != self.out_features:
            proj_x = torch.permute(inputs, [0, 2, 1])
            proj_x = torch.permute(self.proj(proj_x), [0, 2, 1])
            x = x + proj_x
        else:
            proj_x = self.proj(x)
            x = x + proj_x

        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, skip_connection={self.skip_connection}, bias={self.bias is not None})"
