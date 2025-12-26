import torch
import torch.nn as nn


class CatEmbedding(nn.Module):
    def __init__(self,
                 num_embeds_features: list[int],
                 embed_dim: int,
                 dropout: float,
                 padding_idx: int = 0,
                 mask_token: bool = True
                 ) -> None:
        super().__init__()
        self.embed = nn.Embedding(sum(num_embeds_features) + int(mask_token),
                                  embed_dim, padding_idx)
        self.norm = nn.RMSNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.pos_embed = nn.Parameter(
            torch.empty(len(num_embeds_features), embed_dim)
        )
        self.offsets = torch.tensor([0] + num_embeds_features[1:])

    def fill_padding_idx_with_zero(self) -> None:
        with torch.no_grad():
            self.embed.weight[self.offsets] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        B - batch size
        T - num tokens (num features)
        D - embedding dim
        :param x: (shape: [B, T]) indexes of features values
        :return: (shape: [B, T, D])
        """
        # эта строка нужна поскольку для разный фичей используются разные диапазоны эмбеддингов в таблице
        # и у нас есть padding_idx когда для фича содержит значение null
        # но сопостовление индексов для категриальных фичей не принимает это во внимание
        # реализация с одним параметром для таблицы эмбеддингок быстрее чем для каждой фичи своя таблица параметрв
        x = self.offsets.bool() * self.offsets + x
        x = self.norm(x + self.pos_embed)
        return self.dropout(x)


# class NumEmbedding(nn.Module):
#     def __init__(self,
#                  n_features: int,
#                  embed_dim: int) -> None:
#         super().__init__()
#         self.weight = nn.Parameter(
#             torch.empty(n_features, 1, embed_dim)
#         )
#         # self.bias = nn.Parameter(
#         #     torch.empty(n_features, embed_dim)
#         # )
#
#     # def reset_parameters(self, init: str, init_args: dict) -> None:
#     #     getattr(nn, init)(self.weight, **init_args)
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
#         x = x @ self.weight
#         # x = x + self.bias
#         return x
#
#
# class HybridEmbedding(nn.Module):
#     def __init__(self,
#                  cat_features_vocab_sizes: List[int],
#                  number_num_features: int,
#                  embed_dim: int,
#                  cat_features_padding_idx: int = 0) -> None:
#         super().__init__()
#         self.cat_embed = CatEmbedding(cat_features_vocab_sizes, embed_dim,
#                                       cat_features_padding_idx)
#         self.num_embed = NumEmbedding(number_num_features, embed_dim)
#         self.pos_embed = nn.Parameter(torch.empty(
#             sum(cat_features_vocab_sizes) + number_num_features,
#             embed_dim
#         ))
#         # self.dropout = nn.Dropout(dropout)
#         self.norm = nn.RMSNorm(embed_dim)
#
#     def reset_parameters(self, init: str, init_args: dict) -> None:
#         self.cat_embed.reset_parameters(init, init_args)
#         self.num_embed.reset_parameters(init, init_args)
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
#         # x = self.norm(x)
#         # return self.dropout(x)


# class SinPosEmbedding(nn.Module):
#     def __init__(self, embed_dim: int, seq_len: int):
#         super().__init__()
#         pe = torch.zeros(seq_len, embed_dim)
#         position = torch.arange(0, seq_len).float().unsqueeze(1)
#
#         # Вычисляем делитель для частот
#         div_term = torch.exp(
#             torch.arange(0, embed_dim, 2).float() *
#             (-torch.tensor(10000.0).log() / embed_dim)
#         )
#
#         # Применяем синус к четным позициям, косинус к нечетным
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#
#         # Регистрируем как буфер (не обучаемый параметр)
#         self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
#
#     def reset_parameters(self, init: str, init_args: dict) -> None:
#         pass
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x + self.pe[:, :x.size(1)]
#
#
# class TrainPosEmbedding(nn.Module):
#     def __init__(self, embed_dim: int, seq_len: int):
#         super().__init__()
#         self.weight = nn.Parameter(torch.empty(seq_len, embed_dim))
#
#     def reset_parameters(self, init: str, init_args: dict) -> None:
#         getattr(nn, init)(self.weight, **init_args)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x + self.weight[:, :x.size(1)]


if __name__ == '__main__':
    # pass
    x = torch.rand(2, 2)
    m = CatEmbedding([2, 2], 2)
    print(m.embed.weight)
    m.fill_padding_idx_with_zero()
    print(m.embed.weight)

