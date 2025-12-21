import math
import torch
import torch.nn as nn

from collections import OrderedDict
import embedding


class Block(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 hidden_dropout: float,
                 dropout: float,
                 act: str) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.attn_norm = nn.RMSNorm(embed_dim)
        self.mlp_norm = nn.RMSNorm(embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.mlp_drop = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          hidden_dropout, False)
        self.mlp = nn.Sequential(OrderedDict(in_proj=nn.Linear(embed_dim, 3 * embed_dim),
                                             act=getattr(nn, act)(),
                                             dropout=nn.Dropout(hidden_dropout),
                                             out_proj=nn.Linear(3 * embed_dim, embed_dim)))

    def _init_module(self,
                     module: nn.Module,
                     init: nn.Module,
                     init_args: dict,
                     out_init_args: dict) -> None:
        for pn, p in module.named_parameters():
            if 'out' in pn:
                init(p, **out_init_args)
            else:
                init(p, **init_args)

    def reset_parameters(self,
                         init: str,
                         init_args: dict,
                         out_init_args: dict) -> None:
        init = getattr(nn, init)
        self._init_module(self.attn, init, init_args, out_init_args)
        self._init_module(self.mlp, init, init_args, out_init_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attn(self.attn_norm(x))
        x = x + self.attn_drop(attn)
        mlp = self.mlp(self.mlp_norm(x))
        x = x + self.mlp_drop(mlp)
        return x


class Transformer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_blocks: int,
                 num_heads: int,
                 act: str,
                 embed_dropout: float,
                 hidden_dropout: float,
                 dropout: float,
                 embed: str,
                 embed_args: dict,
                 pos_embed: str,
                 init: str,
                 init_args: dict) -> None:
        super().__init__()
        self.embed = getattr(embedding, embed)(embed_dim, **embed_args)
        self.pe = getattr(embedding, pos_embed)(embed_dim)
        self.embed_drop = nn.Dropout(embed_dropout)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, hidden_dropout, dropout, act) for _ in range(num_blocks)
        ])
        self.reset_parameters(init, init_args)

    def reset_parameters(self,
                         init: str,
                         init_args: dict) -> None:
        out_init_args = {k: v / math.sqrt(2 * len(self.blocks)) for k, v in init_args.items()}
        self.embed.reset_parameters(init, init_args, out_init_args)
        self.pe.reset_parameters(init, init_args, out_init_args)
        for block in self.blocks:
            block.reset_parameters(init, init_args, out_init_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed_drop(self.embed(x) + self.pe)
        for block in self.blocks:
            x = block(x)
        return x
