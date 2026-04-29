import rtdl_num_embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F


# class FeatureTokenizer(nn.Module):
#     def __init__(self,
#                  n_num_features: int,
#                  n_cat_features: list[int],
#                  embed_dim: int,
#                  dropout: float,
#                  add_cls_token: bool,
#                  ) -> None:
#         super().__init__()
#         self.n_num_features = n_num_features
#         self.register_buffer('n_cat_features', torch.tensor(n_cat_features))
#         self.register_buffer('offsets', torch.tensor([0] + n_cat_features[:-1]).cumsum(0))
#         self.add_cls_token = add_cls_token
#         self.mask_idx = sum(n_cat_features)
#         self.seq_len = len(num_embed_features) + add_cls_token
#
#         self.weight = nn.Parameter(torch.empty(len(n_cat_features), embed_dim))
#         self.bias = nn.Parameter(torch.empty(self.seq_len, embed_dim))
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
#         assert torch.all(x < self.num_embed_features)
#         x = x + self.offsets
#         if self.add_cls_token:
#             x = torch.cat(
#                 [
#                     torch.tensor([[self.mask_idx]] * x.size(0), device=x.device),
#                     x
#                 ],
#                 dim=1
#             )
#         if mask is not None:
#             x = x.masked_fill(mask, self.mask_idx)
#
#         x = F.embedding(x, self.weight)
#         x = x + self.bias
#         x = self.dropout(x)
#         return x


class FeatureTokenizerEnsemble(nn.Module):
    def __init__(self,
                 embed_dim,
                 n_embed_num: int | list[int],
                 n_embed_cat: list[int],
                 k: int,
                 num_act: str,
                 dropout: float,
                 add_cls_token: bool,
                 ) -> None:
        super().__init__()
        self.k = k
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.empty(1, k, 1, embed_dim))
        else:
            self.register_parameter('cls_token', None)

        if isinstance(n_embed_num, int):
            self.n_num = n_embed_num
            self.num_weight = nn.Parameter(torch.empty(k, self.n_num, 2 * embed_dim))
            self.num_bias = nn.Parameter(torch.empty(k, self.n_num, 2 * embed_dim))
            self.num_act = getattr(nn, num_act)()
        else:
            self.n_num = 0
            self.register_parameter('num_weight', None)
            n_embed_cat = n_embed_num + n_embed_cat

        self.n_cat = len(n_embed_cat)
        self.register_buffer('n_embed_cat_features',
                             torch.tensor(n_embed_cat))
        self.seq_len = self.n_num + self.n_cat + int(add_cls_token)

        offsets = torch.tensor([0] + n_embed_cat[:-1]).cumsum(0).unsqueeze(0)
        self.register_buffer('offsets', torch.cat([offsets + i * sum(n_embed_cat) for i in range(k)]))
        self.cat_weight = nn.Parameter(torch.empty(k * sum(n_embed_cat), embed_dim))

        self.bias = nn.Parameter(torch.empty(k, self.seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # self.num_embed = nn.ModuleList([
        #     rtdl_num_embeddings.PeriodicEmbeddings(
        #         n_num, embed_dim,
        #         n_frequencies=2*embed_dim,
        #         lite=False
        #     )
        #     for _ in range(k)
        # ])

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        if self.num_weight is None:
            cat = torch.cat([num, cat], dim=1)
            num = None
        else:
            num = (
                num
                .reshape(-1, 1, self.n_num, 1)
                .repeat(1, self.k, 1, 1)
            )
            num = num * self.num_weight + self.num_bias
            x_num_1, x_num_2 = num.chunk(2, -1)
            num = self.num_act(x_num_1) * x_num_2

        assert torch.all(cat < self.n_embed_cat_features)

        cat = (
            cat
            .reshape(-1, 1, self.n_cat)
            .repeat(1, self.k, 1)
        )
        cat = cat + self.offsets
        cat = F.embedding(cat, self.cat_weight)

        if self.cls_token is None:
            x = [cat] if num is None else [num, cat]
        else:
            if num is None:
                x = [self.cls_token.repeat(cat.size(0), 1, 1, 1), cat]
            else:
                x = [self.cls_token.repeat(cat.size(0), 1, 1, 1), num, cat]

        x = torch.cat(x, dim=2)
        x = x + self.bias
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    em = FeatureTokenizerEnsemble(
        4,
        2,
        [2, 2],
        3,
        'ReLU',
        0,
        True
    )

    t = torch.cat(
        [torch.randint(0, 2, (1, 1)), torch.randint(0, 2, (1, 1))],
        dim=-1
    )
    print(t)
    print(
        em(
            torch.rand(1, 2),
            t
        ).shape
    )
