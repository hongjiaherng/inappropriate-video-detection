import torch
import torch.nn as nn

from models.pengwu_net import GraphConvolution

gc = GraphConvolution(
    3,
    4,
    skip_connection=True,
    bias=False
)
linear = nn.Linear(3, 4, bias=True)

inputs = torch.rand(2, 5, 3)
adj_mat = torch.rand(2, 5, 5)

print(gc(inputs, adj_mat).shape)
print(linear(inputs).shape)

for k, v in gc.named_parameters():
    print(f"{k}: {v.shape}")

for k, v in linear.named_parameters():
    print(f"{k}: {v.shape}")

print(gc)
