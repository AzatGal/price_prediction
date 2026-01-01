import torch
import torch.nn as nn

from transformer import Transformer


# class MaskTableModeling(Transformer):
#     def __init__(self,
#                  embed_dim: int,
#                  num_blocks: int,
#                  num_heads: int,
#                  hidden_act: str,
#                  num_embeds_features: list[int],
#                  embed_dropout: float,
#                  hidden_dropout: float,
#                  dropout: float,
#                  init: str,
#                  init_args: dict
#                  ) -> None:
#         self.norm = nn.RMSNorm(embed_dim)
#         self.tm_head = nn.Linear(embed_dim, sum(num_embeds_features))
#         self.mask_token = nn.Parameter(torch.empty(embed_dim))
#         super().__init__(embed_dim,
#                          num_blocks,
#                          num_heads,
#                          hidden_act,
#                          num_embeds_features,
#                          embed_dropout,
#                          hidden_dropout,
#                          dropout,
#                          init,
#                          init_args)
#
#     def _embed(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         x = self.embed(x)
#         x[mask] = self.mask_token
#         x = x + self.pos_embed
#         x = self.embed_norm(x)
#         x = self.embed_drop(x)
#         return x
#
#     def _head(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         x = x[mask]
#         x = self.norm(x)
#         x = self.tm_head(x)
#         return x
#
#     def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         x = self._embed(x, mask)
#         x = self.blocks(x)
#         x = self._head(x, mask)
#         return x


class PricePredictor(Transformer):
    def __init__(self,
                 embed_dim: int,
                 num_blocks: int,
                 num_heads: int,
                 hidden_act: str,
                 num_embeds_features: list[int],
                 embed_dropout: float,
                 hidden_dropout: float,
                 dropout: float,
                 init: str,
                 init_args: dict,
                 ) -> None:
        self.norm = nn.RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, sum(num_embeds_features))
        super().__init__(embed_dim,
                         num_blocks,
                         num_heads,
                         hidden_act,
                         num_embeds_features,
                         embed_dropout,
                         hidden_dropout,
                         dropout,
                         init,
                         init_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._embed(x)
        x = self.blocks(x)[:, 0]
        x = self.norm(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    for n, m in m.named_modules():
        print(n, m)