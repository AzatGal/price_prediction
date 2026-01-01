import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from embedding import CatEmbedding


class SparseAttentionV1(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 ratio_tokens_selection: float,
                 num_tokens: int,
                 bias: bool = False,
                 ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        assert 0 < ratio_tokens_selection < 1
        self.num_tokens = num_tokens
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.token_scorer = nn.Sequential(
            nn.Linear(embed_dim, self.num_heads),
            nn.Tanh()
        )
        self.top_k_tokens = int(ratio_tokens_selection * num_tokens)

        self.dropout = dropout
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)

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
        # tanh + 1 должно не сильно изменять серднее и отклонение значений
        q = q * (0.01*token_scores.unsqueeze(3) + 1)
        x = F.scaled_dot_product_attention(
            q, k.gather(2, indices), v.gather(2, indices),
            dropout_p=self.dropout if self.training else 0.0
        )
        x = (x
             .transpose(1, 2)
             .reshape(-1, self.num_tokens, self.embed_dim))
        x = self.out_proj(x)
        return x, token_scores


class SparseAttentionV2(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 ratio_token_selection: float,
                 num_tokens: int,
                 temp: float = 0.01,
                 bias: bool = False,
                 ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        assert 0 < ratio_token_selection < 1

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_tokens = num_tokens
        self.ratio_token_selection = 1 - ratio_token_selection
        self.scale = embed_dim ** 0.5
        self.temp = temp
        self.num_heads = num_heads

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.sel_proj = nn.Linear(num_tokens, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        q, k, v = (self.in_proj(x)
                   .reshape(-1, self.num_tokens, self.num_heads,
                            3, self.head_dim)
                   .transpose(1, 2)
                   .unbind(3))
        x = torch.einsum('bhij,bhkj->bhki', q, k)
                # (q @ k.transpose(2, 3)) / self.scale

        mask = self.sel_proj(x.transpose(2, 3)) / self.temp
        mask = F.sigmoid(mask) > self.ratio_token_selection
        mask = mask.transpose(2, 3)

        x = x.masked_fill(mask, float('-inf'))
        x = self.dropout(x)
        x = x.softmax(3) @ v
        x = x.transpose(1, 2).reshape(-1, self.num_tokens, self.embed_dim)
        x = self.out_proj(x)
        return x, None


# class SparseAttentionV3(nn.Module):
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
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, bias)
#
#     def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
#         x = self.com_proj(x.transpose(1, 2)).transpose(1, 2)
#         x, _ = self.attn(x, x, x)
#         x = self.uncom_proj(x.transpose(1, 2)).transpose(1, 2)
#         return x, None
#
#
class Attention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 ratio_token_selection: float,
                 num_tokens: int,
                 temp: float = 0.01,
                 bias: bool = False,
                 ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        # assert 0 < ratio_token_selection < 1

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_tokens = num_tokens
        self.ratio_token_selection = 1 - ratio_token_selection
        self.scale = embed_dim ** 0.5
        self.temp = temp
        self.num_heads = num_heads

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.sel_proj = nn.Linear(num_tokens, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        q, k, v = (self.in_proj(x)
                   .reshape(-1, self.num_tokens, self.num_heads,
                            3, self.head_dim)
                   .transpose(1, 2)
                   .unbind(3))
        x = torch.einsum('bhij,bhkj->bhki', q, k)
                # (q @ k.transpose(2, 3)) / self.scale

        x = self.dropout(x)
        x = x.softmax(3) @ v
        x = x.transpose(1, 2).reshape(-1, self.num_tokens, self.embed_dim)
        x = self.out_proj(x)
        return x, None


class Block(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 hidden_dropout: float,
                 dropout: float,
                 hidden_act: str,
                 ratio_token_selection: float,
                 hidden_dim: int,
                 num_tokens: int,
                 bias: bool = False
                 ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.attn_norm = nn.RMSNorm(embed_dim)
        self.mlp_norm = nn.RMSNorm(embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.mlp_drop = nn.Dropout(dropout)
        if ratio_token_selection < 0:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                              hidden_dropout, False)
            self.attn_forward = lambda x: self.attn(x, x, x)[0]
            # self.attn = Attention(embed_dim, num_heads, hidden_dropout,
            #                       ratio_token_selection, num_tokens)
            # self.attn_forward = lambda x: self.attn(x)[0]
        else:
            self.attn = SparseAttentionV1(embed_dim, num_heads, hidden_dropout,
                                          ratio_token_selection, num_tokens)
            self.attn_forward = lambda x: self.attn(x)[0]
        self.mlp = nn.Sequential(OrderedDict(in_proj=nn.Linear(embed_dim, hidden_dim, bias),
                                             act=getattr(nn, hidden_act)(),
                                             dropout=nn.Dropout(hidden_dropout),
                                             out_proj=nn.Linear(hidden_dim, embed_dim, bias)))

    def _attn_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_norm(x)
        x = self.attn_forward(x)
        return self.attn_drop(x)

    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp_norm(x)
        x = self.mlp(x)
        return self.mlp_drop(x)

    def forward(self, x: torch.Tensor, pool: bool = False) -> torch.Tensor:
        x = x + self._attn_block(x)
        if pool:
            x = x[:, 0]
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
                 ratio_token_selection: float,
                 hidden_dim: int,
                 num_tokens: int,
                 # init: str,
                 # init_args: dict,
                 ) -> None:
        super().__init__()
        self.embed = CatEmbedding(num_embeds_features, embed_dim, embed_dropout)
        self.norm = nn.RMSNorm(embed_dim)
        self.get_pool = lambda i: i == (num_blocks - 1)
        self.blocks = nn.ModuleList(*[
            Block(embed_dim, num_heads, hidden_dropout, dropout,
                  hidden_act, ratio_token_selection, hidden_dim, num_tokens)
            for _ in range(num_blocks)
        ])
        self.head = nn.Linear(embed_dim, 1)
        # self.decay_params, self.no_decay_params = self.reset_parameters(init, init_args)
        # self.embed.zero_offsets()

    def configure_optimizers(self, init_cfg: dict) -> torch.optim.Optimizer:
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
        self.embed.zero_offsets()
        return decay, no_decay

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(x, mask)
        for i, block in enumerate(self.blocks):
            x = block(x, self.get_pool(i))
        x = self.norm(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    # attn1 = SparseAttention(
    #     embed_dim=3,
    #     num_heads=1,
    #     dropout=0.0,  # Без dropout для детерминированности
    #     top_k_tokens=2
    # ).eval()
    # attn2 = nn.MultiheadAttention(
    #     embed_dim=3,
    #     num_heads=1,
    #     dropout=0.0,  # Без dropout для детерминированности
    #     bias=False,
    #     batch_first=True
    # ).eval()
    # # attn2.in_proj_weight = attn1.in_proj.weight
    # # attn2.out_proj.weight = attn1.out_proj.weight
    #
    # # print(attn2.out_proj.weight)
    # # print()
    # # print(attn1.out_proj.weight)
    # t = torch.randn(128, 16, 3)
    # print(t)
    # print()
    # with torch.no_grad():
    #     t1, _ = attn1(t)
    #     print(t1.mean(), t1.std())
    #     print()
    #     t2, _ = attn2(t, t, t)
    #     print(t2.mean(), t2.std())

    import time

    B, T, C = 2, 3, 4
    size = 1
    t1 = [0] * size
    t2 = [0] * size

    for i in range(size):
        m = torch.randn(B, T, C)
        print(m)
        print()
        attn = Block(
            embed_dim=C,
            num_heads=2,
            hidden_dropout=0.0,
            dropout=0.0,
            hidden_act='ReLU',
            ratio_token_selection=-1,
            hidden_dim=4,
            num_tokens=T
        )
        t = time.time()
        a = attn(m)
        print(a)
        print()
        t2[i] = time.time() - t

        attn = Block(
            embed_dim=C,
            num_heads=2,
            hidden_dropout=0.0,
            dropout=0.0,
            hidden_act='ReLU',
            ratio_token_selection=0.5,  #
            hidden_dim=4,
            num_tokens=T
        )
        t = time.time()
        a = attn(m)
        print(a)
        print()
        t1[i] = time.time() - t


    print(sum(t1) / len(t1))
    print(sum(t2) / len(t2))

