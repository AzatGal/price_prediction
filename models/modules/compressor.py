import torch
import torch.nn as nn


class Compressor(nn.Module):
    def __init__(self,
                 seq_len: int,
                 dim_factor: float,
                 act: str,
                 dropout: float
                 ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(seq_len, round(dim_factor * seq_len))
        self.act = getattr(nn, act)()
        self.out_proj = nn.Linear(round(dim_factor * seq_len), 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class GatedCompressor(nn.Module):
    def __init__(self,
                 seq_len: int,
                 dim_factor: float,
                 act: str,
                 dropout: float
                 ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(seq_len, 2 * round(dim_factor * seq_len))
        self.act = getattr(nn, act)()
        self.out_proj = nn.Linear(round(dim_factor * seq_len), 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = self.in_proj(x).chunk(2, dim=-1)
        x = self.act(x) * y
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
