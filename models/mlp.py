import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 dim_factor: float,
                 dropout: float,
                 act: str,
                 bias: bool = False
                 ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, int(dim_factor * embed_dim), bias)
        self.out_proj = nn.Linear(int(dim_factor * embed_dim), embed_dim, bias)
        self.dropout = nn.Dropout(dropout)
        self.act = getattr(nn, act)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class GLUMLP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 dim_factor: float,
                 dropout: float,
                 act: str,
                 bias: bool = False
                 ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, 2 * int(dim_factor * embed_dim), bias)
        self.out_proj = nn.Linear(int(dim_factor * embed_dim), embed_dim, bias)
        self.dropout = nn.Dropout(dropout)
        self.act = getattr(nn, act)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = self.in_proj(x).chunk(2, dim=-1)
        x = self.act(x) * y
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
