import torch
import torch.nn as nn

from models.modules.attention import AttentionEnsemble
from models.modules.mlp import GatedMLPEnsemble
from models.modules.norm import NormEnsemble


# class TransformerBlock(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  num_q_heads: int,
#                  num_kv_heads: int,
#                  attn_dropout: float,
#                  mlp_dropout: float,
#                  dropout: float,
#                  act: str,
#                  mlp_dim_factor: float,
#                  attn: str,
#                  mlp: str,
#                  norm: str,
#                  ) -> None:
#         super().__init__()
#         self.attn_norm = getattr(nn, norm)(embed_dim)
#         self.mlp_norm = getattr(nn, norm)(embed_dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.mlp_drop = nn.Dropout(dropout)
#
#         self.mlp = getattr(mlp_obj, mlp)(embed_dim, mlp_dim_factor, mlp_dropout, act)
#         self.attn = getattr(attn_obj, attn)(embed_dim, num_q_heads, num_kv_heads, attn_dropout)
#
#     def _attn_block(self,
#                     x: torch.Tensor,
#                     kv_compressors: nn.ModuleList | nn.Linear = None,
#                     mask: torch.Tensor = None
#                     ) -> torch.Tensor:
#         x = self.attn_norm(x)
#         x = self.attn(x, kv_compressors, mask)
#         x = self.attn_drop(x)
#         return x
#
#     def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.mlp_norm(x)
#         x = self.mlp(x)
#         x = self.mlp_drop(x)
#         return x
#
#     def forward(self,
#                 x: torch.Tensor,
#                 kv_compressors: nn.ModuleList | nn.Linear = None,
#                 mask: torch.Tensor = None
#                 ) -> torch.Tensor:
#         x = x + self._attn_block(x, kv_compressors, mask)
#         x = x + self._mlp_block(x)
#         return x


class TransformerEnsembleBlock(nn.Module):
    def __init__(self,
                 seq_len: int,
                 kv_compression_dim: int,
                 attn_bias: bool,
                 attn_dropout: float,
                 mlp_dim_factor: float,
                 act: str,
                 mlp_dropout: float,
                 mlp_bias: bool,
                 embed_dim: int,
                 dropout: float,
                 k: int,
                 share_weights: bool,
                 # add_cls_token: bool
                 ) -> None:
        super().__init__()
        self.attn_norm = NormEnsemble('RMSNorm', embed_dim, k)
        self.attn = AttentionEnsemble(embed_dim, seq_len, kv_compression_dim, attn_dropout,
                                      k, share_weights, attn_bias)
        self.attn_drop = nn.Dropout(dropout)

        self.mlp_norm = NormEnsemble('RMSNorm', embed_dim, k)
        self.mlp = GatedMLPEnsemble(embed_dim, mlp_dim_factor, act, mlp_dropout, k,
                                    share_weights, mlp_bias)
        self.mlp_drop = nn.Dropout(dropout)

    def _attn_block(self, x: torch.Tensor,
                    y: torch.Tensor,
                    # cls_token_only_attn: bool
                    ) -> torch.Tensor:
        x = self.attn_norm(x)
        x = self.attn(x,
                      y,
                      # cls_token_only_attn
                      )
        x = self.attn_drop(x)
        return x

    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.mlp_drop(x)
        return x

    def forward(self, x: torch.Tensor,
                y: torch.Tensor,
                # cls_token_only_attn: bool
                ) -> torch.Tensor:
        x = x + self._attn_block(x,
                                 y,
                                 # cls_token_only_attn
                                 )
        x = x + self._mlp_block(x)
        return x


if __name__ == '__main__':
    pass
    # embed = FeatureEmbedding([2, 2], 3, 0)
    # print(embed.embed.weight)
    # example = torch.randint(0, 2, (2, 2))
    # mask = torch.zeros(2, 2).bool()
    # mask[:, 0] = True
    # # print(fi)
    # # print()
    # t = embed(example, mask)
