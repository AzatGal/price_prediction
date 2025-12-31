from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from embedding import CatEmbedding


class SparseAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 top_k_tokens: int,
                 bias: bool = False,
                 ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.token_scorer = nn.Sequential(
            nn.Linear(embed_dim, self.num_heads),
            nn.Tanh()
        )
        self.top_k_tokens = top_k_tokens

        self.dropout = dropout
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        B, T, C = x.shape

        q, k, v = (self.in_proj(x)
                   .reshape(B, T, self.num_heads, 3, self.head_dim)
                   .transpose(1, 2)
                   .unbind(3))
        token_scores = (self.token_scorer(x)
                        .transpose(1, 2)
                        .unsqueeze(3)
                        .repeat(1, 1, 1, self.head_dim))
        # if self.training:
        #     token_scores = (token_scores + torch.randn_like(token_scores)) / 2
        token_scores = 0.01*token_scores + 1
        indices = (token_scores
                   .topk(self.top_k_tokens, 2, sorted=False)
                   .indices)
        q = q * token_scores  # tanh + 1 должно не сильно изменять серднее и отклонение значений
        x = F.scaled_dot_product_attention(
            q, k.gather(2, indices), v.gather(2, indices),
            dropout_p=self.dropout if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.out_proj(x)
        return x, token_scores


class Block(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 hidden_dropout: float,
                 dropout: float,
                 hidden_act: str,
                 top_k_tokens: int,
                 bias: bool = False
                 ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.attn_norm = nn.RMSNorm(embed_dim)
        self.mlp_norm = nn.RMSNorm(embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.mlp_drop = nn.Dropout(dropout)
        if top_k_tokens < 0:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                              hidden_dropout, False)
            self.attn_forward = lambda x: self.attn(x, x, x)[0]
        else:
            self.attn = SparseAttention(embed_dim, num_heads,
                                        hidden_dropout, top_k_tokens)
            self.attn_forward = lambda x: self.attn(x)[0]
        self.mlp = nn.Sequential(OrderedDict(in_proj=nn.Linear(embed_dim, 4 * embed_dim, bias),
                                             act=getattr(nn, hidden_act)(),
                                             dropout=nn.Dropout(hidden_dropout),
                                             out_proj=nn.Linear(4 * embed_dim, embed_dim, bias)))

    def _attn_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_norm(x)
        x = self.attn_forward(x)
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
                 embed_dropout: float,
                 hidden_dropout: float,
                 dropout: float,
                 top_k_tokens: int,
                 init: str,
                 init_args: dict,
                 ) -> None:
        super().__init__()
        self.embed = CatEmbedding(num_embeds_features, embed_dim)
        self.pos_embed = nn.Parameter(torch.empty(sum(num_embeds_features), embed_dim))
        self.norm = nn.RMSNorm(embed_dim)
        self.embed_drop = nn.Dropout(embed_dropout)
        self.embed_norm = nn.RMSNorm(embed_dim)
        self.blocks = nn.ModuleList(*[
            Block(embed_dim, num_heads, hidden_dropout, dropout, hidden_act, top_k_tokens)
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
                if 'bias' in n:
                    nn.init.zeros_(p)
                else:
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

    def _embed(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)

        x = x + self.pos_embed
        x = self.embed_norm(x)
        x = self.embed_drop(x)
        return x


if __name__ == "__main__":
    attn1 = SparseAttention(
        embed_dim=3,
        num_heads=1,
        dropout=0.0,  # Без dropout для детерминированности
        top_k_tokens=2
    ).eval()
    attn2 = nn.MultiheadAttention(
        embed_dim=3,
        num_heads=1,
        dropout=0.0,  # Без dropout для детерминированности
        bias=False,
        batch_first=True
    ).eval()
    # attn2.in_proj_weight = attn1.in_proj.weight
    # attn2.out_proj.weight = attn1.out_proj.weight

    # print(attn2.out_proj.weight)
    # print()
    # print(attn1.out_proj.weight)
    t = torch.randn(128, 16, 3)
    print(t)
    print()
    with torch.no_grad():
        t1, _ = attn1(t)
        print(t1.mean(), t1.std())
        print()
        t2, _ = attn2(t, t, t)
        print(t2.mean(), t2.std())

    # attn = Block(
    #     embed_dim=4,
    #     num_heads=2,
    #     hidden_dropout=0.0,
    #     dropout=0.0,
    #     hidden_act='ReLU',
    #     top_k_tokens=3
    # )
    # t = torch.randn(2, 8, 4)
    # print(attn(t).shape)
