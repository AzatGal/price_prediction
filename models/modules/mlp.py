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
        self.weight = nn.Parameter(torch.empty(k, in_features, out_features))  # k,
        self.register_parameter(
            'bias', nn.Parameter(torch.empty(k, 1, out_features)) if bias else None,
        )
        # self.r = nn.Parameter(torch.empty(k, 1, in_features))
        # self.s = nn.Parameter(torch.empty(k, 1, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x * self.r
        x = x @ self.weight
        # x = x * self.s
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
        hidden_dim = 2 * round(dim_factor * embed_dim)
        self.in_proj = nn.Linear(embed_dim, hidden_dim, bias)
        self.out_proj = nn.Linear(round(dim_factor * embed_dim), embed_dim, bias)

        self.in_r = nn.Parameter(torch.empty(k, 1, embed_dim))
        self.in_s = nn.Parameter(torch.empty(k, 1, hidden_dim))
        self.out_r = nn.Parameter(torch.empty(k, 1, embed_dim))
        self.out_s = nn.Parameter(torch.empty(k, 1, embed_dim))

        # self.in_proj = LinearEnsemble(embed_dim, 2 * round(dim_factor * embed_dim), k, bias)
        # self.out_proj = LinearEnsemble(round(dim_factor * embed_dim), embed_dim, k, bias)
        self.dropout = nn.Dropout(dropout)
        self.act = getattr(nn, act)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.in_r
        x = self.in_proj(x)
        x = x * self.in_s

        x, y = x.chunk(2, dim=-1)
        x = self.act(x) * y

        x = self.dropout(x)
        x = x * self.out_r
        x = self.out_proj(x)
        x = x * self.out_s
        return x
