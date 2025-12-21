import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CatEmbedding(nn.Module):
    def __init__(self,
                 features_vocab_size: List[int],
                 embed_dim: int,
                 padding_idx: int,
                 dropout: float,
                 bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(sum(features_vocab_size), embed_dim)
        )
        self.bias = nn.Parameter(
            torch.empty(len(features_vocab_size), embed_dim)
        ) if bias else False
        self.padding_idx = padding_idx
        self.offsets = torch.tensor(features_vocab_size)
        self.dropout = nn.Dropout(dropout)

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
        x = F.embedding(x, self.weight, self.padding_idx)
        if self.bias:
            x = x + self.bias
        return self.dropout(x)


class NumEmbedding(nn.Module):
    def __init__(self,
                 num_features: int,
                 embed_dim: int,
                 dropout: float,
                 bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_features, 1, embed_dim)
        )
        self.bias = nn.Parameter(
            torch.empty(num_features, embed_dim)
        ) if bias else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        B - batch size
        T - num tokens (num features)
        D - embedding dim
        :param x: (shape: [B, T]) numbers of features values
        :return: (shape: [B, T, D])
        """
        x = x.unsqueeze(2)
        x = x @ self.weight
        if self.bias:
            x = x + self.bias
        return self.dropout(x)


class HybridEmbedding(nn.Module):
    def __init__(self,
                 cat_features_vocab_size: List[int],
                 num_num_features: int,
                 embed_dim: int,
                 cat_features_padding_idx: int,
                 dropout: float,
                 bias: bool = False) -> None:
        super().__init__()
        self.cat_embed = CatEmbedding(cat_features_vocab_size, embed_dim,
                                      cat_features_padding_idx, dropout, bias)
        self.num_embed = NumEmbedding(num_num_features, embed_dim,
                                      dropout, bias)

    def forward(self, cat_x: torch.Tensor, num_x: torch.Tensor) -> torch.Tensor:
        """
        B - batch size
        T - num tokens (num features)
        D - embedding dim
        :param cat_x: (shape: [B, T]) indexes of categorial features values
        :param num_x: (shape: [B, T]) values of numeric features
        :return: (shape: [B, T, D])
        """
        cat_x = self.cat_embed(cat_x)
        num_x = self.num_embed(num_x)
        x = torch.cat([cat_x, num_x], dim=2)
        return x


if __name__ == '__main__':
    x = torch.rand(2, 2)
    print(x)
    m = NumEmbedding(2, 3)
    print(m(x).shape)

