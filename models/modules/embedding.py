import rtdl_num_embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.mlp import LinearEnsemble
from models.modules.norm import NormEnsemble


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
                 dropout: float,
                 # add_cls_token: bool,
                 share_weights: bool = True,
                 ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.empty(1, k, 1, embed_dim))

        self.share_weights = share_weights
        k_ = 1 if share_weights else k

        if isinstance(n_embed_num, int):
            self.n_num = n_embed_num

            # self.num_weight_1 = nn.Parameter(torch.empty(k_, self.n_num, embed_dim // 2))
            # self.num_weight_2 = nn.Parameter(torch.empty(k_, self.n_num, embed_dim, embed_dim))
            # self.num_bias = nn.Parameter(torch.empty(k_, self.n_num, embed_dim // 2))

            self.num_weight = nn.Parameter(torch.empty(k_, self.n_num, embed_dim))
            self.num_bias = nn.Parameter(torch.empty(k_, self.n_num, embed_dim))
        else:
            self.n_num = 0
            self.register_buffer(
                'n_embed_num', torch.tensor(n_embed_num)
            )
            self.register_parameter('num_weight', None)
            n_embed_cat = n_embed_num + n_embed_cat

        self.n_cat = len(n_embed_cat)
        self.register_buffer(
            'n_embed_cat', torch.tensor(n_embed_cat)
        )
        self.seq_len = self.n_num + self.n_cat # + int(add_cls_token)

        offsets = torch.tensor([0] + n_embed_cat[:-1]).cumsum(0).unsqueeze(0)
        self.register_buffer(
            'offsets', torch.cat([offsets + i * sum(n_embed_cat) for i in range(k_)])
        )
        self.cat_weight = nn.Parameter(torch.empty(k_ * sum(n_embed_cat), embed_dim))

        self.bias = nn.Parameter(torch.empty(k, self.seq_len, embed_dim))
        self.norm = NormEnsemble('RMSNorm', embed_dim, k)
        self.dropout = nn.Dropout(dropout)

        if self.share_weights:
            self.rank = nn.Parameter(torch.randn(k, self.seq_len, embed_dim))
            # nn.init.normal_(self.rank)

        # self.num_embed = rtdl_num_embeddings.PeriodicEmbeddings(self.n_num, embed_dim, lite=False)
        #     nn.ModuleList([
        #     rtdl_num_embeddings.PeriodicEmbeddings(self.n_num, embed_dim, lite=False)
        #     for _ in range(k_)
        # ]))

    def forward(self,
                x_num: torch.Tensor,
                x_cat: torch.Tensor = None
                ) -> (torch.Tensor, torch.Tensor):
        if self.n_num == 0:
            x_cat = x_num if x_cat is None else torch.cat([x_num, x_cat], dim=1)
            x_num = None
        else:
            x_num = x_num.reshape(-1, 1, self.n_num, 1)
            x_num = x_num * self.num_weight + self.num_bias
            # print(x_num.shape)

            # x_num = x_num * self.num_weight_1 + self.num_bias
            # x_num = torch.cat([torch.sin(x_num), torch.cos(x_num)], dim=-1)
            # x_num = F.relu(x_num.unsqueeze(-2) @ self.num_weight_2).squeeze(-2)
            # x_num_1, x_num_2 = x_num.chunk(2, dim=-1)
            # x_num = x_num_1 - x_num_2

            # x_num = x_num.reshape(-1, 1, self.n_num, 1, 1)
            # x_num = F.relu(x_num @ self.num_weight1 + self.num_bias) @ self.num_weight2
            # x_num = x_num.squeeze(-2)

        if x_cat is None:
            x_ = x_num
        else:
            assert torch.all(x_cat < self.n_embed_cat)

            x_cat = x_cat.reshape(-1, 1, self.n_cat)
            x_cat = x_cat + self.offsets
            x_cat = F.embedding(x_cat, self.cat_weight)

            if x_num is None:
                x_ = x_cat
            else:
                x_ = torch.cat([x_num, x_cat], dim=2)

            # if self.cls_token is None:
            #     x = [x_cat] if x_num is None else [x_num, x_cat]
            # else:
            #     if x_num is None:
            #         x = [self.cls_token.repeat(x_cat.size(0), 1, 1, 1), x_cat]
            #     else:
            #         x = [self.cls_token.repeat(x_cat.size(0), 1, 1, 1), x_num, x_cat]
            # for i in x:
            #     print(i.shape)
            # x = torch.cat(x, dim=2)

        if self.share_weights:
            x_ = x_ * self.rank
        x_ = x_ + self.bias
        x_ = self.norm(x_)
        # x_ = self.dropout(x_)

        x = self.cls_token.repeat(x_.size(0), 1, 1, 1)  # * self.rank
        return x, x_


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
