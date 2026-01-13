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
                 pred_dim: int,
                 log_softmax: bool = False,
                 compression_factor: float = None,
                 compression: str = None,
                 ) -> None:
        super().__init__()
        self.embed = FeatureEmbedding(num_embed_features, embed_dim, dropout)

        self.log_softmax = log_softmax
        self.seq_len = self.embed.seq_len
        self.compression_factor = compression_factor
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

    def last_hidden_states(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embed(x, mask)
        for i, block in enumerate(self.blocks):
            x = block(x, mask)
        return x

    def _get_compressors(self, same: bool = True) -> (nn.Linear, nn.Linear):
        c = nn.Linear(self.seq_len, max(1, int(self.compression_factor * self.seq_len)), False)
        if same:
            return c, c
        else:
            return c, nn.Linear(self.seq_len, max(1, int(self.compression_factor * self.seq_len)), False)

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
                if 'bias' in pn:
                    nn.init.zeros_(p)
                else:
                    if 'head' in pn:
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


class MaskedTableAutoencoder(Transformer):
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
                 pred_dim: int,
                 log_softmax: bool = False,
                 compression_factor: float = None,
                 compression: str = None,
                 ) -> None:
        super().__init__(embed_dim, num_embed_features, num_heads, attn_dropout, mlp_dropout,
                         dropout, act, mlp_dim_factor, num_blocks, attn, mlp, norm, pred_dim,
                         log_softmax, compression_factor, compression)
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
            [mask_x, unmask_x], dim=1
        ).gather(
            1, torch.cat([mask_ids, unmask_ids], dim=1).argsort(1)
        )

        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed

        for i, block in enumerate(self.decoder_blocks):
            x = block(x, None)

        x = x[mask].reshape(x.size(0), -1, x.size(2))
        x = self.decoder_norm(x)
        x = self.decoder_head(x)

        if self.log_softmax:
            x = F.log_softmax(x, -1)
        return x

    def configure_optimizer(self,
                            lr: float,
                            weight_decay: float,
                            lr_decay_by_block: float = None,
                            **optim_kwargs
                            ) -> torch.optim.Optimizer:
        return super().configure_optimizer(lr, weight_decay, lr_decay_by_block,
                                           betas=(0.9, 0.95))


class MaskedTableModeling(Transformer):
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
                 pred_dim: int,
                 log_softmax: bool = False,
                 compression_factor: float = None,
                 compression: str = None,
                 ) -> None:
        super().__init__(embed_dim, num_embed_features, num_heads, attn_dropout, mlp_dropout,
                         dropout, act, mlp_dim_factor, num_blocks, attn, mlp, norm, pred_dim,
                         log_softmax, compression_factor, compression)
        self.tm_head = nn.Linear(embed_dim, pred_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.last_hidden_states(x, mask)
        x = x[mask].reshape(x.size(0), -1, x.size(2))
        x = self.norm(x)
        x = self.tm_head(x)
        if self.log_softmax:
            x = F.log_softmax(x, -1)
        return x

    def configure_optimizer(self,
                            lr: float,
                            weight_decay: float,
                            lr_decay_by_block: float = None,
                            **optim_kwargs
                            ) -> torch.optim.Optimizer:
        return super().configure_optimizer(lr, weight_decay, lr_decay_by_block,
                                           betas=(0.9, 0.95))


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
                 pred_dim: int,
                 log_softmax: bool = False,
                 compression_factor: float = None,
                 compression: str = None,
                 ) -> None:
        super().__init__(embed_dim, num_embed_features, num_heads, attn_dropout, mlp_dropout,
                         dropout, act, mlp_dim_factor, num_blocks, attn, mlp, norm, pred_dim,
                         log_softmax, compression_factor, compression)
        self.pp_head = nn.Linear(embed_dim, pred_dim)
        # self.pp_head = nn.Sequential(nn.SiLU(),
        #                              nn.Linear(embed_dim, pred_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.last_hidden_states(x, None)
        x = x.mean(1)
        x = self.norm(x)
        x = self.pp_head(x)
        if self.log_softmax:
            x = F.log_softmax(x, -1)
        return x


if __name__ == '__main__':
    from configs.model_cfg import cfg
    opt = Transformer(**cfg).configure_optimizer(0.1, 0.1)
