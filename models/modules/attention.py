import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_q_heads: int,
                 num_kv_heads: int,
                 dropout: float,
                 # kv_compressors: nn.ModuleList | nn.Linear = None,
                 bias: bool = False
                 ) -> None:
        super().__init__()
        assert embed_dim % num_q_heads == 0
        assert num_q_heads % num_kv_heads == 0

        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_q_heads
        self.dropout = dropout

        self.split_size = [embed_dim] + [self.head_dim * num_kv_heads] * 2

        self.qkv_proj = nn.Linear(embed_dim, sum(self.split_size), bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)

        # self.kv_compressors = kv_compressors

    def _reshape_by_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, _, seq_len = mask.shape
        ids = (torch.arange(seq_len, device=x.device)
               .reshape(1, 1, seq_len, 1)
               .repeat(B, self.num_kv_heads, -1, self.head_dim)[mask]
               .reshape(B, self.num_kv_heads, -1, self.head_dim))
        x = torch.zeros(
            B, self.num_kv_heads, seq_len, self.head_dim,
            dtype=x.dtype, device=x.device
        ).scatter_add_(2, ids, x)
        return x

    def forward(self,
                x: torch.Tensor,
                kv_compressors: nn.ModuleList | nn.Linear = None,
                mask: torch.Tensor = None
                ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x).split(self.split_size, 2)
        qkv = [
            x.reshape(B, T, -1, self.head_dim).transpose(1, 2)
            for x in qkv
        ]
        if kv_compressors is not None:
            if mask is not None:
                # mask - не стандартная маска внимания, а маска видимых токенов у MaskedTableAutoencoder
                mask = mask.unsqueeze(1).repeat(-1, self.num_kv_heads, -1)
                qkv = [qkv[0]] + [self._reshape_by_mask(x, mask) for x in qkv[1:]]
            if isinstance(kv_compressors, nn.ModuleList):
                qkv = [qkv[0]] + [
                    kv_compressors[i](x.transpose(2, 3)).transpose(2, 3)
                    for i, x in enumerate(qkv[1:])
                ]
            else:
                qkv = [qkv[0]] + [
                    kv_compressors(x.transpose(2, 3)).transpose(2, 3)
                    for x in qkv[1:]
                ]

        # q, k, v = qkv
        # # print(q.shape)
        # # print(k.repeat(1, 2, 1, 1).transpose(2, 3).shape)
        # w = (q @ k.repeat_interleave(2, dim=1).transpose(2, 3)) / math.sqrt(self.head_dim)
        # # print('w', w[0].squeeze())
        # w = F.softmax(w, dim=-1)
        # # print('sm', w[0].squeeze())
        # a = F.dropout(w, self.dropout, self.training) @ v.repeat_interleave(2, dim=1)
        # # print('v', v[0].squeeze())
        # # print('a', a[0].squeeze())
        # # a = F.scaled_dot_product_attention(
        # #     *qkv,
        # #     dropout_p=self.dropout if self.training else 0.0,
        # #     enable_gqa=True
        # # )
        a = self.out_proj(
            # a
            qkv[2].repeat_interleave(2, dim=1)
            .transpose(1, 2)
            .reshape(B, -1, C)
        )
        return a


# class GlobalPooling(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  seq_len: int,
#                  dropout: float,
#                  bias: bool = False
#                  ) -> None:
#         super().__init__()
#         self.seq_len = seq_len
#         # self.compressor = nn.Linear(seq_len, 1)
#         self.compressor = nn.Conv1d(seq_len, 1, 1)
#         self.in_proj = nn.Linear(embed_dim, embed_dim, bias)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x = self.in_proj(x)
#         # x = self.dropout(x)
#         print(x.shape)
#         x = self.compressor(x)
#         print(x.shape)
#         # x = self.out_proj(x)
#         return x


if __name__ == '__main__':
    pass
