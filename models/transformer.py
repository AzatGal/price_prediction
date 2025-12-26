import math
from typing import List, Optional

import torch
import torch.nn as nn

from collections import OrderedDict
from embedding import HybridEmbedding, CatEmbedding


class Block(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 hidden_dropout: float,
                 dropout: float,
                 hidden_act: str) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.attn_norm = nn.RMSNorm(embed_dim)
        self.mlp_norm = nn.RMSNorm(embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.mlp_drop = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          hidden_dropout, False)
        self.mlp = nn.Sequential(OrderedDict(in_proj=nn.Linear(embed_dim, 4 * embed_dim),
                                             act=getattr(nn, hidden_act)(),
                                             dropout=nn.Dropout(hidden_dropout),
                                             out_proj=nn.Linear(4 * embed_dim, embed_dim)))

    # def _init_module(self,
    #                  module: nn.Module,
    #                  init: nn.Module,
    #                  init_args: dict,
    #                  out_init_args: dict) -> None:
    #     for pn, p in module.named_parameters():
    #         if 'out' in pn:
    #             init(p, **out_init_args)
    #         else:
    #             init(p, **init_args)
    #
    # def reset_parameters(self,
    #                      init: str,
    #                      init_args: dict,
    #                      out_init_args: dict) -> None:
    #     init = getattr(nn, init)
    #     self._init_module(self.attn, init, init_args, out_init_args)
    #     self._init_module(self.mlp, init, init_args, out_init_args)

    def _attn_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_norm(x)
        x = self.attn(x)
        return self.attn_drop(x)

    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp_norm(x)
        x = self.mlp(x)
        return self.mlp_drop(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attn_block(x)
        x = x + self._mlp_block(x)
        return x


class Transformer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_blocks: int,
                 num_heads: int,
                 hidden_act: str,
                 num_embeds_features: List[int],
                 # cat_features_num_embeds: List[int],
                 # number_num_features: int,
                 embed_dropout: float,
                 hidden_dropout: float,
                 dropout: float,
                 init: str,
                 init_args: dict,
                 padding_idx: int = 0,
                 mask_token: bool = True
                 ) -> None:
        super().__init__()
        # self.embed = HybridEmbedding(cat_features_num_embeds, number_num_features,
        #                              embed_dim, padding_idx)
        self.embed = CatEmbedding(num_embeds_features, embed_dim,
                                  embed_dropout, padding_idx, mask_token)
        self.norm = nn.RMSNorm(embed_dim)
        # self.embed_drop = nn.Dropout(embed_dropout)
        # self.embed_norm = nn.RMSNorm(embed_dim)
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, hidden_dropout, dropout, hidden_act)
            for _ in range(num_blocks)
        ])
        self.decay_params, self.no_decay_params = self.reset_parameters(init, init_args)
        self.embed.fill_padding_idx_with_zero()

    def reset_parameters(self,
                         init: str,
                         init_args: dict) -> (list, list):
        init = getattr(nn, init)
        decay = set()
        no_decay = set()
        params = {n: p for n, p in self.named_parameters()}
        for n, p in params.items():
            if 'norm' not in n:
                init(p, **init_args)
            if any(t in n for t in ('embed', 'norm', 'bias')):
                no_decay.add(n)
            else:
                decay.add(n)
        assert len(decay & no_decay) == 0
        assert len(params.keys() - decay | no_decay) == 0
        decay = [params[n] for n in list(decay)]
        no_decay = [params[n] for n in list(no_decay)]
        return decay, no_decay

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.blocks(x)
        return self.norm(x)


if __name__ == '__main__':
    pass
    # m = Transformer(2,
    #                 2,
    #                 2,
    #                 'ReLU',
    #                 0.0,
    #                 0.0,
    #                 0.0,
    #                 'CatEmbedding',
    #                 )
