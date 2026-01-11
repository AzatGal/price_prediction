import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mlp import GLUMLP


class FeatureEmbedding(nn.Module):
    def __init__(self,
                 num_embed_features: list[int],
                 embed_dim: int,
                 norm: str,
                 dropout: float
                 ) -> None:
        super().__init__()
        # nn.Embedding
        # self.num_embed_features = torch.tensor(num_embed_features)
        self.num_embeds = sum(num_embed_features)
        # self.mask_token = nn.Parameter(torch)
        self.weight = nn.Parameter(torch.empty(sum(num_embed_features) + 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.empty(len(num_embed_features), embed_dim))
        # self.register_buffer(
        #     'offsets',
        #     torch.tensor(num_embed_features).cumsum(0)
        # )
        # self.norm = getattr(nn, norm)(embed_dim)
        self.dropout = nn.Dropout(dropout)

    # @torch.no_grad()
    # def fill_last_values_features_zero(self):
    #     self.weight[self.num_embeds] = 0

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            x = x.masked_fill(mask, self.num_embeds)
            # i = torch.randint(0, mask.size(1), (1,)).item()
            # x[torch.arange(x.size(0)), i] = self.num_embeds
        x = F.embedding(x, self.weight)
        x = x + self.pos_embed
        # x = self.norm(x)
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    em = FeatureEmbedding([4, 2], 2, '', 0.1)
    nn.init.normal_(em.weight, std=0.02)
    em.fill_last_values_features_zero()
    print(em.weight)
