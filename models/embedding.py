import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.mlp import GLUMLP


class FeatureEmbedding(nn.Module):
    def __init__(self,
                 num_embed_features: list[int],
                 embed_dim: int,
                 dropout: float
                 ) -> None:
        super().__init__()
        self.num_embed_features = torch.tensor(num_embed_features)
        # print(self.num_embed_features)
        self.mask_idx = sum(num_embed_features)
        self.seq_len = len(num_embed_features)
        self.weight = nn.Parameter(torch.empty(self.mask_idx + 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.empty(self.seq_len, embed_dim))
        self.register_buffer(
            'offsets',
            torch.tensor([0] + num_embed_features[:-1]).cumsum(0)
        )
        # print(self.offsets)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        assert torch.all(x < self.num_embed_features)
        x = x + self.offsets
        if mask is not None:
            x = x.masked_fill(mask, self.mask_idx)
        # print(x)
        # print(x)
        x = F.embedding(x, self.weight)
        x = x + self.pos_embed
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    em = FeatureEmbedding([4, 2], 2, '', 0.1)
    nn.init.normal_(em.weight, std=0.02)
    em.fill_last_values_features_zero()
    print(em.weight)
