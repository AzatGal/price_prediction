import torch
import torch.nn as nn

import models.modules.attention as attn_obj
import models.modules.mlp as mlp_obj
import models.modules.compressor as compressor_obj
from models.modules.norm import NormEnsemble


class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_q_heads: int,
                 num_kv_heads: int,
                 attn_dropout: float,
                 mlp_dropout: float,
                 dropout: float,
                 act: str,
                 mlp_dim_factor: float,
                 attn: str,
                 mlp: str,
                 norm: str,
                 ) -> None:
        super().__init__()
        self.attn_norm = getattr(nn, norm)(embed_dim)
        self.mlp_norm = getattr(nn, norm)(embed_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.mlp_drop = nn.Dropout(dropout)

        self.mlp = getattr(mlp_obj, mlp)(embed_dim, mlp_dim_factor, mlp_dropout, act)
        self.attn = getattr(attn_obj, attn)(embed_dim, num_q_heads, num_kv_heads, attn_dropout)

    def _attn_block(self,
                    x: torch.Tensor,
                    kv_compressors: nn.ModuleList | nn.Linear = None,
                    mask: torch.Tensor = None
                    ) -> torch.Tensor:
        x = self.attn_norm(x)
        x = self.attn(x, kv_compressors, mask)
        x = self.attn_drop(x)
        return x

    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.mlp_drop(x)
        return x

    def forward(self,
                x: torch.Tensor,
                kv_compressors: nn.ModuleList | nn.Linear = None,
                mask: torch.Tensor = None
                ) -> torch.Tensor:
        x = x + self._attn_block(x, kv_compressors, mask)
        x = x + self._mlp_block(x)
        return x


class TransformerBlockEnsemble(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 k: int,
                 attn_dropout: float,
                 mlp_dropout: float,
                 dropout: float,
                 act: str,
                 mlp_dim_factor: float,
                 attn: str,
                 mlp: str,
                 norm: str,
                 ) -> None:
        super().__init__()
        # self.attn_norm = getattr(nn, norm)(embed_dim, elementwise_affine=False)
        # self.mlp_norm = getattr(nn, norm)(embed_dim, elementwise_affine=False)
        self.attn_norm = NormEnsemble(norm, embed_dim, k)
        self.mlp_norm = NormEnsemble(norm, embed_dim, k)

        self.attn_drop = nn.Dropout(dropout)
        self.mlp_drop = nn.Dropout(dropout)

        self.mlp = getattr(mlp_obj, mlp)(embed_dim, mlp_dim_factor, k, act, mlp_dropout)
        self.attn = getattr(attn_obj, attn)(embed_dim, k, attn_dropout)

    def _attn_block(self,
                    x: torch.Tensor,
                    cls_token_only_attn: bool,
                    kv_compressors: nn.ModuleList | nn.Linear = None,
                    mask: torch.Tensor = None
                    ) -> torch.Tensor:
        x = self.attn_norm(x)
        x = self.attn(x, cls_token_only_attn, kv_compressors, mask)
        x = self.attn_drop(x)
        return x

    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.mlp_drop(x)
        return x

    def forward(self,
                x: torch.Tensor,
                cls_token_only_attn: bool,
                kv_compressors: nn.ModuleList | nn.Linear = None,
                mask: torch.Tensor = None
                ) -> torch.Tensor:
        x = x + self._attn_block(x, cls_token_only_attn, kv_compressors, mask)
        x = x + self._mlp_block(x)
        return x


class CompressorBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 seq_len: int,
                 # comp_dropout: float,
                 mlp_dropout: float,
                 dropout: float,
                 act: str,
                 # comp_dim_factor: float,
                 mlp_dim_factor: float,
                 compressor: str,
                 mlp: str,
                 norm: str,
                 ) -> None:
        super().__init__()
        self.comp_norm = getattr(nn, norm)(embed_dim)
        self.mlp_norm = getattr(nn, norm)(embed_dim)

        self.comp_drop = nn.Dropout(dropout)
        self.mlp_drop = nn.Dropout(dropout)

        self.mlp = getattr(mlp_obj, mlp)(embed_dim, mlp_dim_factor, mlp_dropout, act)
        # self.compressor = nn.Linear(seq_len, 1)
        self.compressor = getattr(compressor_obj, compressor)(embed_dim, seq_len)  # comp_dim_factor, act, comp_dropout

    def _compressor_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.comp_norm(x)
        x = self.compressor(x)  # .transpose(1, 2)
        x = self.comp_drop(x)
        return x

    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.mlp_drop(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._compressor_block(x)
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
