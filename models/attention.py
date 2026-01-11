import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectedAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 k_compressor: nn.Linear = None,
                 v_compressor: nn.Linear = None,
                 bias: bool = False
                 ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.num_tokens = 20

        self.token_scorer = nn.Sequential(
            nn.Linear(embed_dim, self.num_heads),
            nn.Tanh()
        )
        self.top_k_tokens = 14

        self.dropout = dropout
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.cls_ids = torch.zeros(
            1, self.num_heads, self.top_k_tokens, self.head_dim,
            dtype=torch.int
        )

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        q, k, v = (self.in_proj(x)
                   .reshape(-1, self.num_tokens,
                            self.num_heads, 3, self.head_dim)
                   .transpose(1, 2)
                   .unbind(3))

        token_scores = self.token_scorer(x).transpose(1, 2)
        indices = (token_scores
                   .topk(self.top_k_tokens, 2, sorted=False)
                   .indices
                   .unsqueeze(3)
                   .expand(-1, -1, -1, self.head_dim))
        indices = torch.cat(
            [
                self.cls_ids.expand(x.size(0), -1, -1, -1), indices
            ], dim=2
        )
        # q = q * (0.01*token_scores.unsqueeze(3) + 1)
        x = F.scaled_dot_product_attention(
            q, k.gather(2, indices), v.gather(2, indices),
            dropout_p=self.dropout if self.training else 0.0
        )
        x = (x
             .transpose(1, 2)
             .reshape(-1, self.num_tokens, self.embed_dim))
        x = self.out_proj(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 k_compressor: nn.Linear = None,
                 v_compressor: nn.Linear = None,
                 bias: bool = False
                 ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)

        self.k_compressor = k_compressor
        self.v_compressor = v_compressor
        self.seq_len = self.k_compressor.in_features
        self.register_buffer('ids', torch.arange(self.seq_len).reshape(1, 1, self.seq_len, 1))
        # .expand(-1, self.num_heads, -1, self.head_dim))

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor = None,
                mask_only_attn: bool = False
                ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = (self.qkv_proj(x)
                   .reshape(B, T, self.num_heads, 3, self.head_dim)
                   .transpose(1, 2)
                   .unbind(3))
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1)
        if mask_only_attn:
            q = q[mask].reshape(B, self.num_heads, -1, self.head_dim)
        if self.k_compressor is not None:
            if T < self.seq_len:
                k = torch.zeros(
                    B, self.num_heads, self.seq_len, self.head_dim,
                    dtype=k.dtype, device=k.device
                ).scatter_(
                    2,
                    (self.ids
                     .expand(B, self.num_heads, -1, self.head_dim)[mask]
                     .reshape(B, self.num_heads, -1, self.head_dim)),
                    k
                )
                # k = t
            k = self.k_compressor(
                k.transpose(2, 3)
            ).transpose(2, 3)
        if self.v_compressor is not None:
            if T < self.seq_len:
                v = torch.zeros(
                    B, self.num_heads, self.seq_len, self.head_dim,
                    dtype=k.dtype, device=k.device
                ).scatter_(
                    2,
                    (self.ids
                     .expand(B, self.num_heads, -1, self.head_dim)[mask]
                     .reshape(B, self.num_heads, -1, self.head_dim)),
                    v
                )
                # v = t
            v = self.v_compressor(
                v.transpose(2, 3)
            ).transpose(2, 3)
        a = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0
        )
        a = self.out_proj(
            a
            .transpose(1, 2)
            .reshape(B, -1, C)
        )
        return a


class LinearAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 k_compressor: nn.Linear = None,
                 v_compressor: nn.Linear = None,
                 bias: bool = False
                 ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads

        self.k_compressor = k_compressor
        self.v_compressor = v_compressor

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = (self.qkv_proj(x)
                   .reshape(B, T, self.num_heads, 3, -1)
                   .transpose(1, 2)
                   .unbind(3))
        k = self.act(k).add_(1.0)
        v = self.act(v).add_(1.0)
        if self.k_compressor is not None:
            k = self.k_compressor(
                k.transpose(2, 3)
            ).transpose(2, 3)
        if self.v_compressor is not None:
            v = self.v_compressor(
                v.transpose(2, 3)
            ).transpose(2, 3)
        kv = k.transpose(2, 3) @ v
        k_sum = (k
                 .sum(dim=2)
                 .unsqueeze(3))
        n = q @ self.dropout(kv)
        d = (q @ k_sum).add_(1e-8)
        a = n.div_(d)
        a = self.out_proj(
            a
            .transpose(1, 2)
            .reshape(B, T, C)
        )
        return a


if __name__ == '__main__':
    example = torch.randn(2, 3, 4)
    attn = LinearAttention(4, 2, 0.0)
    print(attn(example).shape)
