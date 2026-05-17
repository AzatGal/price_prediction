import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.mlp import LinearEnsemble
from models.modules.norm import NormEnsemble


# class Attention(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  num_q_heads: int,
#                  num_kv_heads: int,
#                  dropout: float,
#                  # kv_compressors: nn.ModuleList | nn.Linear = None,
#                  bias: bool = False
#                  ) -> None:
#         super().__init__()
#         assert embed_dim % num_q_heads == 0
#         assert num_q_heads % num_kv_heads == 0
#
#         self.embed_dim = embed_dim
#         self.num_q_heads = num_q_heads
#         self.num_kv_heads = num_kv_heads
#         self.head_dim = embed_dim // num_q_heads
#         self.dropout = dropout
#
#         self.split_size = [embed_dim] + [self.head_dim * num_kv_heads] * 2
#
#         self.qkv_proj = nn.Linear(embed_dim, sum(self.split_size), bias)
#         # self.q_proj = nn.Linear(embed_dim, embed_dim, bias)
#         # self.k_proj = nn.Conv1d(19, 6*embed_dim, embed_dim)
#         # self.v_proj = nn.Conv1d(19, 6*embed_dim, embed_dim)
#
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias)
#
#         # self.compressor = nn.Linear(19, 6)
#         # self.uncompressor = nn.Linear(6, 19)
#
#         # self.v_biases = nn.Parameter(torch.zeros(1, 19, 1, self.head_dim))
#
#     def _reshape_by_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         B, _, seq_len = mask.shape
#         ids = (torch.arange(seq_len, device=x.device)
#                .reshape(1, 1, seq_len, 1)
#                .repeat(B, self.num_kv_heads, -1, self.head_dim)[mask]
#                .reshape(B, self.num_kv_heads, -1, self.head_dim))
#         x = torch.zeros(
#             B, self.num_kv_heads, seq_len, self.head_dim,
#             dtype=x.dtype, device=x.device
#         ).scatter_add_(2, ids, x)
#         return x
#
#     def forward(self,
#                 x: torch.Tensor,
#                 kv_compressors: nn.ModuleList | nn.Linear = None,
#                 mask: torch.Tensor = None
#                 ) -> torch.Tensor:
#         # x = self.compressor(
#         #     x.transpose(1, 2)
#         # ).transpose(1, 2)
#         B, T, C = x.shape
#
#         # q = (self.q_proj(x)
#         #      .reshape(B, T, self.num_q_heads, self.head_dim)
#         #      .transpose(1, 2))
#         # k = (self.k_proj(x)
#         #      .reshape(B, -1, self.num_q_heads, self.head_dim)
#         #      .transpose(1, 2))
#         # v = (self.v_proj(x)
#         #      .reshape(B, -1, self.num_q_heads, self.head_dim)
#         #      .transpose(1, 2))
#         qkv = self.qkv_proj(x).split(self.split_size, 2)
#         qkv = [
#             x.reshape(B, T, -1, self.head_dim).transpose(1, 2)
#             for x in qkv
#         ]
#
#         if kv_compressors is not None:
#             # qkv[1] = qkv[1] + self.k_biases
#             # qkv[2] = qkv[2] + self.v_biases
#             # qkv[1] = F.relu(qkv[1])
#             # qkv[1:] = [
#             #     F.relu(x) for x in qkv[1:]
#             # ]
#
#             if mask is not None:
#                 # mask - не стандартная маска внимания, а маска видимых токенов у MaskedTableAutoencoder
#                 mask = mask.unsqueeze(1).repeat(-1, self.num_kv_heads, -1)
#                 qkv[1:] = [self._reshape_by_mask(x, mask) for x in qkv[1:]]
#             if isinstance(kv_compressors, nn.ModuleList):
#                 # qkv[1] = kv_compressors[0](qkv[1].transpose(2, 3)).transpose(2, 3)
#                 qkv[1:] = [
#                     kv_compressors[i](x.transpose(2, 3)).transpose(2, 3)
#                     for i, x in enumerate(qkv[1:])
#                 ]
#             else:
#                 # qkv[1] = kv_compressors(qkv[1].transpose(2, 3)).transpose(2, 3)
#                 qkv[1:] = [
#                     kv_compressors(x.transpose(2, 3)).transpose(2, 3)
#                     for x in qkv[1:]
#                 ]
#
#         # qkv[1:] = [
#         #     x.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
#         #     for x in qkv[1:]
#         # ]
#         # q, k, v = qkv
#         # w = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)
#         #
#         # w = F.sigmoid(w).repeat(1, 1, 1, T)  # > 0.5).float()
#         # a = F.dropout(w, self.dropout, self.training) @ v
#
#         # a = w.repeat(1, 1, 1, T) @ v
#
#         # qkv[0] = qkv[0][:, :, :1]
#
#         a = F.scaled_dot_product_attention(
#             *qkv,
#             # q, k, v,
#             dropout_p=self.dropout if self.training else 0.0,
#             enable_gqa=True,
#             # scale=10 / math.sqrt(self.head_dim)
#         )
#         # a = F.dropout(qkv[2].repeat(1, 1, T, 1), self.dropout, self.training)
#
#         # a = qkv[0] * qkv[2].repeat(1, 1, T, 1)
#
#         # t = self.compressor(qkv[2].transpose(2, 3)).transpose(2, 3)
#         # a = qkv[0] * t  # .repeat(1, 1, T, 1)
#
#         a = self.out_proj(
#             a
#             # qkv[2]
#             # .repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
#             .transpose(1, 2)
#             .reshape(B, -1, C)
#         )
#
#         # a = torch.cat(
#         #     [
#         #         a,
#         #         torch.zeros(B, T - 1, C, device=a.device)
#         #     ],
#         #     dim=1
#         # )
#
#         # a = self.uncompressor(
#         #     a.transpose(1, 2)
#         # ).transpose(1, 2)
#         return a


class AttentionEnsemble(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 seq_len: int,
                 kv_compression_dim: int,
                 dropout: float,
                 k: int,
                 share_weights: bool,
                 # add_cls_token: bool,
                 bias: bool
                 ) -> None:
        super().__init__()
        self.dropout = dropout
        # k_ = 1 if share_weights else k
        # self.register_buffer('mask', torch.ones(k_, seq_len, 1))
        # with torch.inference_mode():
        #     self.mask[:, int(add_cls_token):].bernoulli_(0.9)

        # self.in_proj = LinearEnsemble(embed_dim, 2 * embed_dim, k, share_weights, bias)
        # self.out_proj = LinearEnsemble(2 * embed_dim, embed_dim, k, share_weights, bias)

        self.k_compressor = LinearEnsemble(seq_len, kv_compression_dim, k, share_weights, bias)
        self.v_compressor = LinearEnsemble(seq_len, kv_compression_dim, k, share_weights, bias)

        self.k_norm = NormEnsemble('RMSNorm', embed_dim, k)
        self.v_norm = NormEnsemble('RMSNorm', embed_dim, k)

    def forward(self,
                # y: torch.Tensor,
                x: torch.Tensor,
                cls_token_only_attn: bool
                ) -> torch.Tensor:
        # x = x * self.mask
        # x = self.in_proj(x)
        k = self.k_norm(self.k_compressor(
            x.transpose(2, 3)
        ).transpose(2, 3))
        v = self.v_norm(self.v_compressor(
            x.transpose(2, 3)
        ).transpose(2, 3))

        if cls_token_only_attn:
            x = x[:, :, :1]

        a = F.scaled_dot_product_attention(
            x, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        # a = self.out_proj(a)
        return a


if __name__ == '__main__':
    pass
