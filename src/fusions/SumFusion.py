import torch
from torch import nn


class SumFusion(nn.Module):
    def __init__(self):
        super(SumFusion, self).__init__()

    def forward(self, embeddings):
        output = torch.stack(embeddings, dim=2).sum(dim=2)
        return output
