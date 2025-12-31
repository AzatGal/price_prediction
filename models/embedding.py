import torch
import torch.nn as nn


class CatEmbedding(nn.Module):
    def __init__(self,
                 num_embeds_features: list[int],
                 embed_dim: int,
                 ) -> None:
        super().__init__()
        self.num_embeds = sum(num_embeds_features)
        self.embed = nn.Embedding(sum(num_embeds_features) + 1,
                                  embed_dim)
        self.offsets = torch.tensor([0] + num_embeds_features[1:]).cumsum(0)

    # def fill_padding_idx_with_zero(self) -> None:
    #     with torch.no_grad():
    #         self.embed.weight[self.offsets] = 0

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # эта строка нужна поскольку для разный фичей используются разные диапазоны эмбеддингов в таблице
        # и у нас есть padding_idx когда для фича содержит значение null
        # но сопостовление индексов для категриальных фичей не принимает это во внимание
        # реализация с одним параметром для таблицы эмбеддингок быстрее чем для каждой фичи своя таблица параметрв
        x = x + self.offsets
        x[mask] = self.num_embeds
        x = self.embed(x)
        return x


# class ConEmbedding(nn.Module):
#     def __init__(self,
#                  num_features: int,
#                  embed_dim: int,
#                  hidden_act: str = None,
#                  in_channels: int = 1
#                  ) -> None:
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_features = num_features
#         if hidden_act is None:
#             self.embed = nn.Conv1d(in_channels, embed_dim, 1,
#                                    groups=num_features, bias=False)
#         else:
#             self.embed = nn.Sequential(
#                 nn.Conv1d(in_channels, 4 * embed_dim, 1,
#                           groups=num_features, bias=False),
#                 getattr(nn, hidden_act),
#                 nn.Conv1d(4 * embed_dim, embed_dim, 1,
#                           groups=num_features, bias=False)
#             )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         B - batch size
#         T - num tokens (num features)
#         D - embedding dim
#         :param x: (shape: [B, T]) numbers of features values
#         :return: (shape: [B, T, D])
#         """
#         x = x.unsqueeze(2)
#         x = self.embed(x)
#         x = x.reshape(-1, self.num_features, self.embed_dim)
#         return x
#
#
# class HybridEmbedding(nn.Module):
#     def __init__(self,
#                  cat_features_vocab_sizes: list[int],
#                  number_num_features: int,
#                  embed_dim: int,
#                  cat_features_padding_idx: int = 0) -> None:
#         super().__init__()
#         self.cat_embed = CatEmbedding(cat_features_vocab_sizes, embed_dim,
#                                       cat_features_padding_idx)
#         self.num_embed = ConEmbedding(number_num_features, embed_dim)
#         self.pos_embed = nn.Parameter(torch.empty(
#             sum(cat_features_vocab_sizes) + number_num_features,
#             embed_dim
#         ))
#         self.norm = nn.RMSNorm(embed_dim)
#
#     def forward(self, num_x: torch.Tensor, cat_x: torch.Tensor) -> torch.Tensor:
#         """
#         B - batch size
#         T - num tokens (num features)
#         D - embedding dim
#         :param cat_x: (shape: [B, T]) indexes of categorial features values
#         :param num_x: (shape: [B, T]) values of numeric features
#         :return: (shape: [B, T, D])
#         """
#         num_x = self.num_embed(num_x)
#         cat_x = self.cat_embed(cat_x)
#         x = torch.cat([num_x, cat_x], dim=2)
#         x = x + self.pos_embed
#         return x


if __name__ == '__main__':
    # pass
    x = (torch.rand(2, 2) * 10).int()
    m = CatEmbedding([10, 10], 2)
    print(m(x).shape)
    print(m.embed.weight)
    m.fill_padding_idx_with_zero()
    print(m.embed.weight)

