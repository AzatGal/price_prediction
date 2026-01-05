import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.attention import LinearAttention, Attention
from models.modules.mlp import GLUMLP, MLP


# class FeatureEmbedding(nn.Module):
#     def __init__(self,
#                  num_embed_features: list[int],
#                  embed_dim: int,
#                  dropout: float,
#                  ) -> None:
#         super().__init__()
#         self.num_embeds = sum(num_embed_features)
#         self.embed = nn.Embedding(sum(num_embed_features) + 1,
#                                   embed_dim)
#         self.pos_embed = nn.Parameter(torch.empty(sum(num_embed_features), embed_dim))
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.RMSNorm(embed_dim)
#         self.register_buffer(
#             'offsets',
#             torch.tensor([[0] + num_embed_features[:-1]]).cumsum(0)
#         )
#
#     def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         """
#         x[:, 0] - target для классификации
#         эта строка нужна поскольку для разный фичей используются разные диапазоны эмбеддингов в таблице
#         и у нас есть padding_idx когда для фича содержит значение null
#         но сопостовление индексов для категриальных фичей не принимает это во внимание
#         реализация с одним параметром для таблицы эмбеддингок быстрее чем для каждой фичи своя таблица параметрв
#         """
#         x = x + self.offsets
#         x = x.masked_fill(mask, self.num_embeds)
#         x = self.embed(x)
#         x = self.norm(x)
#         x = self.dropout(x)
#         return x


# class SparseAttention(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  num_heads: int,
#                  dropout: float,
#                  ratio_tokens_selection: float,
#                  num_tokens: int,
#                  bias: bool = False,
#                  ) -> None:
#         super().__init__()
#         assert embed_dim % num_heads == 0
#         assert 0 < ratio_tokens_selection < 1
#         self.num_tokens = num_tokens
#         self.head_dim = embed_dim // num_heads
#         self.num_heads = num_heads
#         self.embed_dim = embed_dim
#
#         self.token_scorer = nn.Sequential(
#             nn.Linear(embed_dim, self.num_heads),
#             nn.Tanh()
#         )
#         self.top_k_tokens = int(ratio_tokens_selection * num_tokens)
#
#         self.dropout = dropout
#         self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias)
#
#     def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
#         q, k, v = (self.in_proj(x)
#                    .reshape(-1, self.num_tokens,
#                             self.num_heads, 3, self.head_dim)
#                    .transpose(1, 2)
#                    .unbind(3))
#         token_scores = self.token_scorer(x).transpose(1, 2)
#         indices = (token_scores
#                    .topk(self.top_k_tokens, 2, sorted=False)
#                    .indices
#                    .unsqueeze(3)
#                    .expand(-1, -1, -1, self.head_dim))
#         # q = q * (0.01*token_scores.unsqueeze(3) + 1)
#         if self.training:
#             token_scores = token_scores + torch.randn_like(token_scores)
#         else:
#             token_scores = token_scores + 1
#         token_scores = 0.01 * token_scores.unsqueeze(3)
#         x = F.scaled_dot_product_attention(
#             q * token_scores, k.gather(2, indices), v.gather(2, indices),
#             dropout_p=self.dropout if self.training else 0.0
#         )
#         x = (x
#              .transpose(1, 2)
#              .reshape(-1, self.num_tokens, self.embed_dim))
#         x = self.out_proj(x)
#         return x, token_scores


# class Attention(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  num_heads: int,
#                  dropout: float,
#                  ratio_token_compression: float,
#                  num_tokens: int,
#                  bias: bool = False,
#                  ) -> None:
#         super().__init__()
#         assert embed_dim % num_heads == 0
#         assert 0 < ratio_token_compression < 1
#
#         t = int(ratio_token_compression * num_tokens)
#         self.com_proj = nn.Linear(num_tokens, t, bias)
#         self.uncom_proj = nn.Linear(t, num_tokens, bias)
#         self.attn = LinearAttention(embed_dim, num_heads, dropout) # , bias)
#
#     def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
#         x = self.com_proj(x.transpose(1, 2)).transpose(1, 2)
#         x, _ = self.attn(x, x, x)
#         x = self.uncom_proj(x.transpose(1, 2)).transpose(1, 2)
#         return x, None
#
#
# """
# Попробовать компрессию на уровне BHTC -> BHtC (t < T)
# """
#
#
# class LinearAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout=0.0, feature_map='elu'):
#         super().__init__()
#         assert embed_dim % num_heads == 0
#
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.dropout = dropout
#         self.feature_map = feature_map
#
#         self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
#         # self.q_proj = nn.Linear(embed_dim, embed_dim)
#         # self.k_proj = nn.Linear(embed_dim, embed_dim)
#         # self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#
#     def _feature_map(self, x):
#         if self.feature_map == 'elu':
#             return F.elu(x) + 1.0
#         elif self.feature_map == 'relu':
#             return F.relu(x) + 1.0
#         else:
#             raise ValueError(f"Unknown feature_map: {self.feature_map}")
#
#     def forward(self, query, key, value, key_padding_mask=None):
#         L, B, E = query.shape
#         S = key.size(0)
#
#         q = self.q_proj(query)  # (L, B, E)
#         k = self.k_proj(key)  # (S, B, E)
#         v = self.v_proj(value)  # (S, B, E)
#
#         # Reshape to (B*H, seq_len, D)
#         q = q.view(L, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3).contiguous().view(B * self.num_heads, L,
#                                                                                               self.head_dim)
#         k = k.view(S, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3).contiguous().view(B * self.num_heads, S,
#                                                                                               self.head_dim)
#         v = v.view(S, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3).contiguous().view(B * self.num_heads, S,
#                                                                                               self.head_dim)
#
#         # Apply feature map
#         q = self._feature_map(q)  # (B*H, L, D)
#         k = self._feature_map(k)  # (B*H, S, D)
#
#         # Handle padding mask
#         if key_padding_mask is not None:
#             # Expand mask to (B*H, S)
#             mask = key_padding_mask.unsqueeze(1).expand(-1, self.num_heads, -1).contiguous().view(B * self.num_heads, S)
#             # Apply mask: set k and v to 0 at padded positions
#             k = k.masked_fill(mask.unsqueeze(-1), 0.0)
#             v = v.masked_fill(mask.unsqueeze(-1), 0.0)
#
#         # Core linear attention computation
#         KV = torch.bmm(k.transpose(-2, -1), v)  # (B*H, D, D)
#
#         # Denominator: Z = Q @ (K^T @ 1) = sum over S
#         K_sum = k.sum(dim=1, keepdim=True)  # (B*H, 1, D)
#         Z = torch.bmm(q, K_sum.transpose(-2, -1)).clamp_min(1e-9)  # (B*H, L, 1)
#
#         # Numerator
#         numerator = torch.bmm(q, KV)  # (B*H, L, D)
#
#         # Output
#         out = numerator / Z  # (B*H, L, D)
#
#         # Reshape back to (L, B, E)
#         out = out.view(B, self.num_heads, L, self.head_dim)
#         out = out.permute(2, 0, 1, 3).contiguous().view(L, B, E)
#
#         out = self.out_proj(out)
#         if self.dropout > 0.0:
#             out = F.dropout(out, p=self.dropout, training=self.training)
#
#         return out, None


class Block(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 attn_dropout: float,
                 mlp_dropout: float,
                 dropout: float,
                 act: str,
                 mlp_dim_factor: float,
                 attn: str,
                 mlp: str,
                 norm: str,
                 k_compressor: nn.Linear = None,
                 v_compressor: nn.Linear = None
                 ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0

        self.attn_norm = getattr(nn, norm)(embed_dim)
        self.mlp_norm = getattr(nn, norm)(embed_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.mlp_drop = nn.Dropout(dropout)

        if mlp == 'GLUMLP':
            self.mlp = GLUMLP(embed_dim, mlp_dim_factor, mlp_dropout, act)
        elif mlp == 'MLP':
            self.mlp = MLP(embed_dim, mlp_dim_factor, mlp_dropout, act)
        else:
            raise NotImplementedError()

        if attn == 'LinearAttn':
            self.attn = LinearAttention(embed_dim, num_heads, attn_dropout, k_compressor, v_compressor)
        elif attn == 'Attn':
            self.attn = Attention(embed_dim, num_heads, attn_dropout, k_compressor, v_compressor)
        else:
            raise NotImplementedError()

    def _attn_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_norm(x)
        x = self.attn(x)
        x = self.attn_drop(x)
        return x

    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.mlp_drop(x)
        return x

    def forward(self, x: torch.Tensor, pool: bool = False) -> torch.Tensor:
        x = x + self._attn_block(x)
        if pool:
            x = x[:, 0]
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
