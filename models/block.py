import torch
import torch.nn as nn

from models.attention import LinearAttention, Attention
from models.mlp import GLUMLP, MLP


class Block(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 attn_dropout: float,
                 mlp_dropout: float,
                 dropout: float,
                 act: str,
                 mlp_dim_factor: float,
                 attn: str,
                 mlp: str,
                 norm: str,
                 k_compressor: nn.Linear = None,
                 v_compressor: nn.Linear = None
                 ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0

        self.attn_norm = getattr(nn, norm)(embed_dim)
        self.mlp_norm = getattr(nn, norm)(embed_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.mlp_drop = nn.Dropout(dropout)

        if mlp == 'GLUMLP':
            self.mlp = GLUMLP(embed_dim, mlp_dim_factor, mlp_dropout, act)
        elif mlp == 'MLP':
            self.mlp = MLP(embed_dim, mlp_dim_factor, mlp_dropout, act)
        else:
            raise NotImplementedError()

        if attn == 'LinearAttn':
            self.attn = LinearAttention(embed_dim, num_heads, attn_dropout, k_compressor, v_compressor)
        elif attn == 'Attn':
            self.attn = Attention(embed_dim, num_heads, attn_dropout, k_compressor, v_compressor)
        else:
            raise NotImplementedError()

    def _attn_block(self, x: torch.Tensor, mask: torch.Tensor, norm: bool) -> torch.Tensor:
        if norm:
            y = self.attn_norm(x)
        else:
            y = x
        y = self.attn(y, mask)
        y = self.attn_drop(y)
        if mask is None:
            y = y + x
        else:
            y = y + x[:, :y.size(1)]
        return y

    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp_norm(x)
        y = self.mlp(y)
        y = self.mlp_drop(y)
        y = y + x
        return y

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, norm: bool = True) -> torch.Tensor:
        x = self._attn_block(x, mask, norm)
        x = self._mlp_block(x)
        return x


if __name__ == '__main__':
    pass
    # embed = FeatureEmbedding([2, 2], 3, 0)
    # print(embed.embed.weight)
    # example = torch.randint(0, 2, (2, 2))
    # mask = torch.zeros(2, 2).bool()
    # mask[:, 0] = True
    # # print(fi)
    # # print()
    # t = embed(example, mask)
