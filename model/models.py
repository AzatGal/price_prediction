from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from transformer import Transformer


class MaskTableModeling(Transformer):
    def __init__(self,
                 embed_dim: int,
                 num_blocks: int,
                 num_heads: int,
                 hidden_act: str,
                 cat_feature_vocab_sizes: List[int],
                 number_num_features: int,
                 embed_dropout: float,
                 hidden_dropout: float,
                 dropout: float,
                 init: str,
                 init_args: dict,
                 mask_ratio: float,
                 cat_features_padding_idx: int = 0) -> None:
        assert 0 < mask_ratio < 1

        self.number_num_features = number_num_features
        self.mask_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        mask = torch.zeros(number_num_features +
                           len(cat_feature_vocab_sizes))
        mask[:mask_ratio].fill_(1)
        self.register_buffer('mask', mask)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(embed_dim)
        self.cat_head = nn.Linear(embed_dim, sum(cat_feature_vocab_sizes))
        self.num_head = nn.Linear(embed_dim, 1)
        super().__init__(embed_dim,
                         num_blocks,
                         num_heads,
                         hidden_act,
                         cat_feature_vocab_sizes,
                         number_num_features,
                         embed_dropout,
                         hidden_dropout,
                         dropout,
                         init,
                         init_args,
                         cat_features_padding_idx)

    def _mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        arg_noise = torch.rand(x.size(0)).argsort()
        mask = self.mask[arg_noise]
        x[mask] = self.mask_token
        return x, mask

    def _pred(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:


    def forward(self, num_x: torch.Tensor, cat_x: torch.Tensor) -> torch.Tensor:
        x = self.embed(num_x, cat_x)
        x, mask = self._mask(x)
        x = self.dropout(self.norm(x))
        x = self.blocks(x)
        return x


if __name__ == '__main__':
    m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    for n, m in m.named_modules():
        print(n, m)