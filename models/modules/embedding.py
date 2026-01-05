import torch
import torch.nn as nn


class FeatureEmbedding(nn.Module):
    def __init__(self,
                 num_embed_features: list[int],
                 embed_dim: int
                 ) -> None:
        super().__init__()
        self.num_embeds = sum(num_embed_features)
        self.embed = nn.Embedding(sum(num_embed_features) + 1,
                                  embed_dim)
        self.pos_embed = nn.Parameter(torch.empty(len(num_embed_features), embed_dim))
        self.register_buffer(
            'offsets',
            torch.tensor([[0] + num_embed_features[:-1]]).cumsum(0)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x[:, 0] - target для классификации
        эта строка нужна поскольку для разный фичей используются разные диапазоны эмбеддингов в таблице
        и у нас есть padding_idx когда для фича содержит значение null
        но сопостовление индексов для категриальных фичей не принимает это во внимание
        реализация с одним параметром для таблицы эмбеддингок быстрее чем для каждой фичи своя таблица параметрв
        """
        x = x + self.offsets
        x = x.masked_fill(mask, self.num_embeds)
        x = self.embed(x)
        x = x + self.pos_embed
        return x
