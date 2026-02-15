import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.embedding import FeatureEmbedding
from models.modules.block import GlobalPoolingBlock
from models.modules.mlp import GatedMLP


class TestModel(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_embed_features: list[int],
                 num_q_heads: int,
                 num_kv_heads: int,
                 attn_dropout: float,
                 mlp_dropout: float,
                 dropout: float,
                 act: str,
                 mlp_dim_factor: float,
                 num_blocks: int,
                 attn: str,
                 mlp: str,
                 norm: str,
                 pool: str,
                 pred_dim: int,
                 add_first_token: bool,
                 mask_first_token: bool,
                 kv_compression: str = None,
                 kv_compression_ratio: float = None,
                 ) -> None:
        super().__init__()
        self.add_first_token = add_first_token
        self.embed = FeatureEmbedding(num_embed_features, embed_dim,
                                      dropout, add_first_token)
        self.seq_len = self.embed.seq_len

        self.blocks = nn.ModuleList([
            GlobalPoolingBlock(embed_dim, self.seq_len, attn_dropout, mlp_dropout, dropout,
                               act, mlp_dim_factor, attn, mlp, norm)
            for _ in range(num_blocks)
        ])
        self.norm = getattr(nn, norm)(embed_dim)

        self.mask_first_token = mask_first_token
        if mask_first_token:
            self.register_buffer('mask', torch.zeros(1, self.seq_len, dtype=torch.bool))
            self.mask[:, 0] = True

        self.tp_head = nn.Linear(embed_dim, pred_dim)

        self.reset_parameters()

    # @torch.no_grad()
    # def zero_compressors_(self):
    #     if self.kv_compressors is not None:
    #         if isinstance(self.kv_compressors, nn.Linear):
    #             self.kv_compressors.weight.zero_()
    #         elif isinstance(self.kv_compressors, nn.ModuleList):
    #             for compressor in self.kv_compressors:
    #                 if isinstance(compressor, nn.Linear):
    #                     compressor.weight.zero_()
    #                 elif isinstance(compressor, nn.ModuleList):
    #                     compressor[0].weight.zero_()
    #                     compressor[1].weight.zero_()
    #                 else:
    #                     raise NotImplementedError()
    #         else:
    #             raise NotImplementedError()

    def reset_parameters(self) -> None:
        for pn, p in self.named_parameters():
            if 'norm' not in pn:
                if 'bias' in pn or 'compressor' in pn:
                    nn.init.zeros_(p)
                elif 'head' in pn:
                    nn.init.kaiming_uniform_(p, a=5 ** 0.5)
                else:
                    nn.init.normal_(p, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask_first_token:
            x = self.embed(x, self.mask)
        else:
            x = self.embed(x)

        for block in self.blocks:
            x = block(x)

        x = x.mean(1)
        x = self.norm(x)
        x = self.tp_head(x)
        return x


if __name__ == '__main__':
    pass

