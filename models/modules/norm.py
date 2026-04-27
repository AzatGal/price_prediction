import torch
import torch.nn as nn
# import torch.nn.functional as F


class RMSNormEnsemble(nn.Module):
    def __init__(self, dim: int, k: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.weight = nn.Parameter(torch.ones(k, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        return self.norm(x) * self.weight
