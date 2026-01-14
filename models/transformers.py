import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embedding import FeatureEmbedding
from models.block import Block


class Transformer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_embed_features: list[int],
                 num_heads: int,
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
                 compression: str = None,
                 compression_ratio: float = None,
                 ) -> None:
        super().__init__()
        self.add_first_token = add_first_token
        self.embed = FeatureEmbedding(num_embed_features, embed_dim,
                                      dropout, add_first_token)
        self.seq_len = self.embed.seq_len

        self.compression_ratio = compression_ratio
        if compression is None:
            compressors = [(None, None)] * num_blocks
        elif compression == 'Head':
            compressors = [self._get_compressors(False) for _ in range(num_blocks)]
        elif compression == 'KV':
            compressors = [self._get_compressors() for _ in range(num_blocks)]
        elif compression == 'Layer':
            compressors = [self._get_compressors()] * num_blocks
        else:
            raise NotImplementedError()

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, attn_dropout, mlp_dropout, dropout, act,
                  mlp_dim_factor, attn, mlp, norm, *compressors[i])
            for i in range(num_blocks)
        ])
        self.norm = getattr(nn, norm)(embed_dim)

    def _get_compressors(self, same: bool = True) -> (nn.Linear, nn.Linear):
        c = nn.Linear(self.seq_len, max(1, int(self.compression_ratio * self.seq_len)), False)
        if same:
            return c, c
        else:
            return c, nn.Linear(self.seq_len, max(1, int(self.compression_ratio * self.seq_len)), False)

    def configure_optimizer(self,
                            lr: float,
                            weight_decay: float,
                            lr_decay_by_block: float = None,
                            **optim_kwargs
                            ) -> torch.optim.Optimizer:
        if lr_decay_by_block is not None:
            lrs = [
                lr * (lr_decay_by_block ** i)
                for i in reversed(
                    range(len(self.blocks) + 2)
                )
            ]
            embed = set()
            norm_blocks = [set() for _ in range(len(self.blocks))]
            blocks = [set() for _ in range(len(self.blocks))]
            norm_head = set()
            head = set()
        else:
            decay = set()
            no_decay = set()

        params = {pn: p for pn, p in self.named_parameters()}
        for pn, p in params.items():
            if 'norm' not in pn:
                if 'bias' in pn or 'compressor' in pn:
                    nn.init.zeros_(p)
                elif 'head' in pn:
                    nn.init.kaiming_uniform_(p, a=5 ** 0.5)
                else:
                    nn.init.normal_(p, std=0.02)
            if lr_decay_by_block is not None:
                if 'embed' in pn:
                    embed.add(pn)
                elif 'blocks' in pn:
                    i = int(pn.split('.')[1])
                    if 'norm' in pn:
                        norm_blocks[i].add(pn)
                    else:
                        blocks[i].add(pn)
                else:
                    if 'norm' in pn:
                        norm_head.add(pn)
                    else:
                        head.add(pn)
            else:
                if any(t in pn for t in ('embed', 'norm', 'bias')):
                    no_decay.add(pn)
                else:
                    decay.add(pn)

        if lr_decay_by_block is not None:
            inter_params = embed & norm_head & head
            for i in range(len(self.blocks)):
                inter_params = inter_params & norm_blocks[i] & blocks[i]

            union_params = embed | norm_head | head
            for i in range(len(self.blocks)):
                union_params = union_params | norm_blocks[i] | blocks[i]

            assert len(inter_params) == 0
            assert len(params.keys() - union_params) == 0

            embed = [params[name] for name in list(embed)]
            for i in range(len(self.blocks)):
                norm_blocks[i] = [params[name] for name in list(norm_blocks[i])]
                blocks[i] = [params[name] for name in list(blocks[i])]
            norm_head = [params[name] for name in list(norm_head)]
            head = [params[name] for name in list(head)]

            optim_groups = [
                {'params': embed, 'lr': lrs[0], 'weight_decay': 0},
                {'params': head, 'lr': lrs[-1], 'weight_decay': weight_decay},
                {'params': norm_head, 'lr': lrs[-1], 'weight_decay': 0},
            ]
            for i in range(len(self.blocks)):
                optim_groups.append({'params': norm_blocks[i], 'lr': lrs[i + 1], 'weight_decay': 0})
                optim_groups.append({'params': blocks[i], 'lr': lrs[i + 1], 'weight_decay': weight_decay})
            return torch.optim.AdamW(optim_groups)
        else:
            assert len(decay & no_decay) == 0
            assert len(params.keys() - (decay | no_decay)) == 0
            decay = [params[name] for name in list(decay)]
            no_decay = [params[name] for name in list(no_decay)]
            optim_groups = [
                {'params': decay, 'weight_decay': weight_decay},
                {'params': no_decay, 'weight_decay': 0.0}
            ]
            return torch.optim.AdamW(optim_groups, lr=lr, **optim_kwargs)


class MaskedTransformer(Transformer):
    def __init__(self,
                 embed_dim: int,
                 num_embed_features: list[int],
                 num_heads: int,
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
                 compression: str = None,
                 compression_ratio: float = None,
                 ) -> None:
        super().__init__(embed_dim, num_embed_features, num_heads, attn_dropout, mlp_dropout,
                         dropout, act, mlp_dim_factor, num_blocks, attn, mlp, norm, pool, pred_dim,
                         add_first_token, mask_first_token, compression, compression_ratio)

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

    def configure_optimizer(self,
                            lr: float,
                            weight_decay: float,
                            lr_decay_by_block: float = None,
                            **optim_kwargs
                            ) -> torch.optim.Optimizer:
        return super().configure_optimizer(lr, weight_decay, lr_decay_by_block,
                                           betas=(0.9, 0.95))


class MaskedTableAutoencoder(MaskedTransformer):
    def __init__(self,
                 embed_dim: int,
                 decoder_embed_dim: int,
                 num_embed_features: list[int],
                 num_heads: int,
                 decoder_num_heads: int,
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
                 compression: str = None,
                 compression_ratio: float = None
                 ) -> None:
        super().__init__(embed_dim, num_embed_features, num_heads, attn_dropout, mlp_dropout,
                         dropout, act, mlp_dim_factor, num_blocks, attn, mlp, norm, pool, pred_dim,
                         add_first_token, mask_first_token, compression, compression_ratio)
        if compression is None:
            compressors = [(None, None)] * num_blocks
        elif compression == 'Head':
            compressors = [self._get_compressors(False) for _ in range(decoder_num_blocks)]
        elif compression == 'KV':
            compressors = [self._get_compressors() for _ in range(decoder_num_blocks)]
        elif compression == 'Layer':
            compressors = [self._get_compressors()] * decoder_num_blocks
        else:
            raise NotImplementedError()

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.empty(self.seq_len, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, attn_dropout, mlp_dropout, dropout,
                  act, mlp_dim_factor, attn, mlp, norm, *compressors[i])
            for i in range(decoder_num_blocks)
        ])
        self.decoder_norm = getattr(nn, norm)(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, pred_dim)
        self.register_buffer('ids', torch.arange(self.seq_len).reshape(1, self.seq_len, 1))

    def forward(self, x: torch.Tensor, mask_ratio: float) -> (torch.Tensor, torch.Tensor):
        mask = self.get_mask(x, mask_ratio)

        x = self.embed(x, mask)
        B, T, C = x.shape

        mask_ids = self.ids.expand(B, -1, C)[mask].reshape(B, -1, C)
        mask_x = x.gather(1, mask_ids)

        unmask_ids = self.ids.expand(B, -1, C)[~mask].reshape(B, -1, C)
        unmask_x = x.gather(1, unmask_ids)

        for i, block in enumerate(self.blocks):
            unmask_x = block(unmask_x, ~mask)
        unmask_x = self.norm(unmask_x)

        x = torch.cat(
            [mask_x, unmask_x], 1
        ).gather(
            1, torch.cat([mask_ids, unmask_ids], dim=1).argsort(1)
        )

        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed

        for i, block in enumerate(self.decoder_blocks):
            x = block(x)

        if self.add_first_token:
            x = x[:, 1:]

        x = x[mask]  # .reshape(x.size(0), -1, x.size(2))
        x = self.decoder_norm(x)
        x = self.decoder_head(x)
        return x, mask


class MaskedTableModeling(MaskedTransformer):
    def __init__(self,
                 embed_dim: int,
                 num_embed_features: list[int],
                 num_heads: int,
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
                 compression: str = None,
                 compression_ratio: float = None,
                 ) -> None:
        super().__init__(embed_dim, num_embed_features, num_heads, attn_dropout, mlp_dropout,
                         dropout, act, mlp_dim_factor, num_blocks, attn, mlp, norm, pool, pred_dim,
                         add_first_token, mask_first_token, compression, compression_ratio)
        self.tm_head = nn.Linear(embed_dim, pred_dim)

    def forward(self, x: torch.Tensor, mask_ratio: float) -> (torch.Tensor, torch.Tensor):
        mask = self.get_mask(x, mask_ratio)

        x = self.embed(x, mask)

        for i, block in enumerate(self.blocks):
            x = block(x)

        x = x[mask]  # .reshape(x.size(0), -1, x.size(2))
        x = self.norm(x)
        x = self.tm_head(x)
        return x, mask


class PricePrediction(Transformer):
    def __init__(self,
                 embed_dim: int,
                 num_embed_features: list[int],
                 num_heads: int,
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
                 compression: str = None,
                 compression_ratio: float = None,
                 ) -> None:
        super().__init__(embed_dim, num_embed_features, num_heads, attn_dropout, mlp_dropout,
                         dropout, act, mlp_dim_factor, num_blocks, attn, mlp, norm, pool, pred_dim,
                         add_first_token, mask_first_token, compression, compression_ratio)
        self.pool = pool
        self.mask_first_token = mask_first_token
        if mask_first_token:
            self.register_buffer('mask', torch.zeros(1, self.seq_len, dtype=torch.bool))
            self.mask[:, 0] = True
        self.pp_head = nn.Linear(embed_dim, pred_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask_first_token:
            x = self.embed(x, self.mask)
        else:
            x = self.embed(x)

        for i, block in enumerate(self.blocks):
            x = block(x)

        if self.pool == 'token':
            x = x[:, 0]
        elif self.pool == 'mean':
            x = x.mean(1)
        else:
            raise NotImplementedError()

        x = self.norm(x)
        x = self.pp_head(x)
        return x


if __name__ == '__main__':
    from configs.model_cfg import cfg
    m = Transformer(**cfg, pred_dim=1)
    o = m.configure_optimizer(0.1, 0.1)
    print(m.embed.pos_embed)
