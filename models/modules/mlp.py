import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 dim_factor: float,
                 act: str,
                 dropout: float,
                 bias: bool = False
                 ) -> None:
        super().__init__()
        hidden_dim = round(dim_factor * embed_dim)
        self.in_proj = nn.Linear(embed_dim, hidden_dim, bias)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias)
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
                 act: str,
                 dropout: float,
                 bias: bool = False
                 ) -> None:
        super().__init__()
        hidden_dim = round(dim_factor * embed_dim)
        self.in_proj = nn.Linear(embed_dim, 2 * hidden_dim, bias)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias)
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
            share_weights: bool,
            bias: bool
    ) -> None:
        super().__init__()
        self.register_parameter(
            'rank', nn.Parameter(torch.ones(k, 1, in_features)) if share_weights else None
        )
        self.register_parameter(
            'scale', nn.Parameter(torch.ones(k, 1, out_features)) if share_weights else None
        )
        self.weight = nn.Parameter(torch.empty(1 if share_weights else k,
                                               in_features, out_features))
        self.register_parameter(
            'bias', nn.Parameter(torch.empty(k, 1, out_features)) if bias else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank is not None:
            x = x * self.rank
        x = x @ self.weight
        if self.scale is not None:
            x = x * self.scale
        if self.bias is not None:
            x = x + self.bias
        return x


class GatedMLPEnsemble(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 dim_factor: float,
                 act: str,
                 dropout: float,
                 k: int,
                 share_weights: bool,
                 bias: bool
                 ) -> None:
        super().__init__()
        hidden_dim = round(dim_factor * embed_dim)

        self.in_proj = LinearEnsemble(embed_dim, 2 * hidden_dim, k, share_weights, bias)
        self.out_proj = LinearEnsemble(hidden_dim, embed_dim, k, share_weights, bias)

        self.dropout = nn.Dropout(dropout)
        self.act = getattr(nn, act)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)

        x, y = x.chunk(2, dim=-1)
        x = self.act(x) * y
        x = self.dropout(x)

        x = self.out_proj(x)
        return x
