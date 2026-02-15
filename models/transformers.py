import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.embedding import FeatureEmbedding
from models.modules.block import TransformerBlock


class Transformer(nn.Module):
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

        self.kv_compression_ratio = kv_compression_ratio
        if kv_compression is None:
            self.kv_compressors = None
        elif kv_compression == 'Head':
            self.kv_compressors = nn.ModuleList([
                nn.ModuleList([self._get_compressor(), self._get_compressor()])
                for _ in range(num_blocks)
            ])
        elif kv_compression == 'KV':
            self.kv_compressors = nn.ModuleList([
                self._get_compressor() for _ in range(num_blocks)
            ])
        elif kv_compression == 'Layer':
            self.kv_compressors = self._get_compressor()
        else:
            raise NotImplementedError()

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_q_heads, num_kv_heads, attn_dropout, mlp_dropout,
                             dropout, act, mlp_dim_factor, attn, mlp, norm)
            for _ in range(num_blocks)
        ])
        self.norm = getattr(nn, norm)(embed_dim)

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

    def _get_compressor(self) -> nn.Linear:
        return nn.Linear(
            self.seq_len,
            max(1, round(self.kv_compression_ratio * self.seq_len)),
            False
        )

    def reset_parameters(self) -> None:
        for pn, p in self.named_parameters():
            if 'norm' not in pn:
                if 'bias' in pn:  # or 'compressor' in pn:
                    nn.init.zeros_(p)
                elif 'head' in pn:
                    nn.init.kaiming_uniform_(p, a=5 ** 0.5)
                else:
                    nn.init.normal_(p, std=0.02)


class MaskedTransformer(Transformer):
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
        super().__init__(embed_dim, num_embed_features, num_q_heads, num_kv_heads, attn_dropout,
                         mlp_dropout, dropout, act, mlp_dim_factor, num_blocks, attn, mlp, norm,
                         pool, pred_dim, add_first_token, mask_first_token,
                         kv_compression, kv_compression_ratio)

    def get_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        B, T = x.shape
        noise = torch.rand(B, T, device=x.device)
        mask = torch.zeros(B, T, device=x.device, dtype=torch.bool)
        mask[:, :int(T * mask_ratio)] = True
        mask = mask.gather(1, noise.argsort(1))
        if self.add_first_token:
            mask = torch.cat(
                [torch.zeros(B, 1, device=x.device, dtype=torch.bool), mask],
                1
            )
        return mask


class MaskedTableAutoencoder(MaskedTransformer):
    def __init__(self,
                 embed_dim: int,
                 decoder_embed_dim: int,
                 num_embed_features: list[int],
                 num_q_heads: int,
                 num_kv_heads: int,
                 decoder_num_q_heads: int,
                 decoder_num_kv_heads: int,
                 attn_dropout: float,
                 mlp_dropout: float,
                 dropout: float,
                 act: str,
                 mlp_dim_factor: float,
                 num_blocks: int,
                 decoder_num_blocks: int,
                 attn: str,
                 mlp: str,
                 norm: str,
                 pool: str,
                 pred_dim: int,
                 add_first_token: bool,
                 mask_first_token: bool,
                 kv_compression: str = None,
                 kv_compression_ratio: float = None
                 ) -> None:
        super().__init__(embed_dim, num_embed_features, num_q_heads, num_kv_heads, attn_dropout,
                         mlp_dropout, dropout, act, mlp_dim_factor, num_blocks, attn, mlp, norm,
                         pool, pred_dim, add_first_token, mask_first_token,
                         kv_compression, kv_compression_ratio)
        if kv_compression is None:
            self.decoder_kv_compressors = None
        elif kv_compression == 'Head':
            self.decoder_kv_compressors = nn.ModuleList([
                nn.ModuleList([self._get_compressor(), self._get_compressor()])
                for _ in range(decoder_num_blocks)
            ])
        elif kv_compression == 'KV':
            self.decoder_kv_compressors = nn.ModuleList([
                self._get_compressors() for _ in range(decoder_num_blocks)
            ])
        elif kv_compression == 'Layer':
            self.decoder_kv_compressors = self._get_compressors()
        else:
            raise NotImplementedError()

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.empty(self.seq_len, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_q_heads, decoder_num_kv_heads, attn_dropout,
                             mlp_dropout, dropout, act, mlp_dim_factor, attn, mlp, norm)
            for _ in range(decoder_num_blocks)
        ])
        self.decoder_norm = getattr(nn, norm)(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, pred_dim)
        self.register_buffer('ids', torch.arange(self.seq_len).reshape(1, self.seq_len, 1))

        self.reset_parameters()

    def forward(self, x: torch.Tensor, mask_ratio: float) -> (torch.Tensor, torch.Tensor):
        mask = self.get_mask(x, mask_ratio)

        x = self.embed(x, mask)
        B, T, C = x.shape

        mask_ids = self.ids.repeat(B, -1, C)[mask].reshape(B, -1, C)
        mask_x = x.gather(1, mask_ids)

        unmask_ids = self.ids.repeat(B, -1, C)[~mask].reshape(B, -1, C)
        unmask_x = x.gather(1, unmask_ids)

        for i, block in enumerate(self.blocks):
            unmask_x = block(
                unmask_x,
                self.kv_compressors[i] if isinstance(self.kv_compressors, nn.ModuleList)
                else self.kv_compressors,
                ~mask
            )
        unmask_x = self.norm(unmask_x)

        x = torch.cat(
            [mask_x, unmask_x], 1
        ).gather(
            1, torch.cat([mask_ids, unmask_ids], dim=1).argsort(1)
        )

        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed

        for i, block in enumerate(self.decoder_blocks):
            x = block(
                x,
                self.decoder_kv_compressors[i] if isinstance(self.decoder_kv_compressors, nn.ModuleList)
                else self.decoder_kv_compressors
            )

        if self.add_first_token:
            x = x[:, 1:]

        x = x[mask]  # .reshape(x.size(0), -1, x.size(2))
        x = self.decoder_norm(x)
        x = self.decoder_head(x)
        return x, mask


class MaskedTableModeler(MaskedTransformer):
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
        super().__init__(embed_dim, num_embed_features, num_q_heads, num_kv_heads, attn_dropout,
                         mlp_dropout, dropout, act, mlp_dim_factor, num_blocks, attn, mlp, norm,
                         pool, pred_dim, add_first_token, mask_first_token,
                         kv_compression, kv_compression_ratio)
        self.tm_head = nn.Linear(embed_dim, pred_dim)

        self.reset_parameters()

    def forward(self, x: torch.Tensor, mask_ratio: float) -> (torch.Tensor, torch.Tensor):
        mask = self.get_mask(x, mask_ratio)

        x = self.embed(x, mask)

        for i, block in enumerate(self.blocks):
            x = block(
                x,
                self.kv_compressors[i] if isinstance(self.kv_compressors, nn.ModuleList) else self.kv_compressors
            )

        x = x[mask]  # .reshape(x.size(0), -1, x.size(2))
        x = self.norm(x)
        x = self.tm_head(x)
        return x, mask


class TablePredictor(Transformer):
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
        super().__init__(embed_dim, num_embed_features, num_q_heads, num_kv_heads, attn_dropout,
                         mlp_dropout, dropout, act, mlp_dim_factor, num_blocks, attn, mlp, norm,
                         pool, pred_dim, add_first_token, mask_first_token,
                         kv_compression, kv_compression_ratio)
        self.pool = pool
        self.mask_first_token = mask_first_token
        if mask_first_token:
            self.register_buffer('mask', torch.zeros(1, self.seq_len, dtype=torch.bool))
            self.mask[:, 0] = True
        self.tp_head = nn.Linear(embed_dim, pred_dim)

        self.reset_parameters()
        if pool == 'w_avg':
            self.w_avg = nn.Parameter(torch.ones(self.seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask_first_token:
            x = self.embed(x, self.mask)
        else:
            x = self.embed(x)

        for i, block in enumerate(self.blocks):
            x = block(
                x,
                self.kv_compressors[i] if isinstance(self.kv_compressors, nn.ModuleList) else self.kv_compressors
            )

        if self.pool == 'cls':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(1)
        elif self.pool == 'sum':
            x = x.sum(1)
        elif self.pool == 'max':
            x = x.max(1).values
        elif self.pool == 'w_avg':
            x = self.w_avg.softmax(0) @ x  # .softmax(0)
        else:
            raise NotImplementedError()

        x = self.norm(x)
        x = self.tp_head(x)
        return x


if __name__ == '__main__':
    from configs.model_cfg import cfg
    # cfg.pop('include_target')
    m = Transformer(**cfg, pred_dim=1)
    o = m.configure_optimizer(0.1, 0.1)
    for k, _ in m.named_parameters():
        print(k)

