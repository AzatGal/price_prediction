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


class GatedMLP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 dim_factor: float,
                 dropout: float,
                 act: str,
                 bias: bool = False
                 ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, 2 * round(dim_factor * embed_dim), bias)
        self.out_proj = nn.Linear(round(dim_factor * embed_dim), embed_dim, bias)
        self.dropout = nn.Dropout(dropout)
        self.act = getattr(nn, act)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = self.in_proj(x).chunk(2, dim=-1)
        x = self.act(x) * y
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LinearEnsemble(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            k: int,
            bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(k, in_features, out_features)
        )
        self.register_parameter(
            'bias',
            nn.Parameter(torch.empty(k, out_features)) if bias else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        # print(self.weight.shape)

        # x = x.transpose(0, 1)
        x = x @ self.weight
        # x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class GatedMLPEnsemble(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 dim_factor: float,
                 k: int,
                 act: str,
                 dropout: float,
                 bias: bool = False
                 ) -> None:
        super().__init__()
        self.k = k
        # self.in_proj = nn.Linear(embed_dim, 2 * round(dim_factor * embed_dim), bias)
        # self.out_proj = nn.Linear(round(dim_factor * embed_dim), embed_dim, bias)
        self.in_proj = LinearEnsemble(embed_dim, 2 * round(dim_factor * embed_dim), k, bias)
        self.out_proj = LinearEnsemble(round(dim_factor * embed_dim), embed_dim, k, bias)
        self.dropout = nn.Dropout(dropout)
        self.act = getattr(nn, act)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = self.in_proj(x).chunk(2, dim=-1)
        x = self.act(x) * y
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
