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
        self.tm_head = nn.Linear(embed_dim, sum(num_embeds_features) + int(mask_token))

        super().__init__(embed_dim,
                         num_blocks,
                         num_heads,
                         hidden_act,
                         num_embeds_features,
                         # cat_features_num_embeds,
                         # number_num_features,
                         embed_dropout,
                         hidden_dropout,
                         dropout,
                         init,
                         init_args,
                         padding_idx,
                         mask_token)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = super()._forward(x)
        x = self.tm_head(x[mask])
        return x


if __name__ == '__main__':
    m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    for n, m in m.named_modules():
        print(n, m)