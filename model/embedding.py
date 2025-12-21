import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CatEmbedding(nn.Module):
    def __init__(self,
                 features_vocab_size: List[int],
                 embed_dim: int,
                 padding_idx: int,
                 # dropout: float,
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
        # self.dropout = nn.Dropout(dropout)

    def reset_parameters(self, init: str, init_args: dict) -> None:
        getattr(nn, init)(self.weight, **init_args)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

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
        return x


class NumEmbedding(nn.Module):
    def __init__(self,
                 num_features: int,
                 embed_dim: int,
                 # dropout: float,
                 bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_features, 1, embed_dim)
        )
        self.bias = nn.Parameter(
            torch.empty(num_features, embed_dim)
        ) if bias else None
        # self.dropout = nn.Dropout(dropout)

    def reset_parameters(self, init: str, init_args: dict) -> None:
        getattr(nn, init)(self.weight, **init_args)

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
        return x


class HybridEmbedding(nn.Module):
    def __init__(self,
                 cat_features_vocab_size: List[int],
                 number_num_features: int,
                 embed_dim: int,
                 cat_features_padding_idx: int,
                 dropout: float,
                 bias: bool = False) -> None:
        super().__init__()
        self.cat_embed = CatEmbedding(cat_features_vocab_size, embed_dim,
                                      cat_features_padding_idx, dropout, bias)
        self.num_embed = NumEmbedding(number_num_features,
                                      embed_dim, dropout, bias)

    def reset_parameters(self, init: str, init_args: dict) -> None:
        self.cat_embed.reset_parameters(init, init_args)
        self.num_embed.reset_parameters(init, init_args)

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


class SinPosEmbedding(nn.Module):
    def __init__(self, embed_dim: int, seq_len: int):
        super().__init__()
        pe = torch.zeros(seq_len, embed_dim)
        position = torch.arange(0, seq_len).float().unsqueeze(1)

        # Вычисляем делитель для частот
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() *
            (-torch.tensor(10000.0).log() / embed_dim)
        )

        # Применяем синус к четным позициям, косинус к нечетным
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Регистрируем как буфер (не обучаемый параметр)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TrainPosEmbedding(nn.Module):
    def __init__(self, embed_dim: int, seq_len: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(seq_len, embed_dim))

    def reset_parameters(self, init: str, init_args: dict) -> None:
        getattr(nn, init)(self.weight, **init_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.weight[:, :x.size(1)]



if __name__ == '__main__':
    x = torch.rand(2, 2)
    print(x)
    m = NumEmbedding(2, 3)
    print(m(x).shape)

