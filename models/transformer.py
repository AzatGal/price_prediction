import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.embedding import FeatureEmbedding
from models.modules.block import Block


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
                 out_log: bool = False,
                 compression_factor: float = None,
                 compression: str = None,
                 ) -> None:
        super().__init__()
        self.embed = FeatureEmbedding(num_embed_features, embed_dim)

        self.out_log = out_log
        self.seq_len = len(num_embed_features)
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

        self.get_pool = lambda i: i == (num_blocks - 1)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, attn_dropout, mlp_dropout, dropout,
                  act, mlp_dim_factor, attn, mlp, norm, *compressors[i])
            for i in range(num_blocks)
        ])

        self.norm = getattr(nn, norm)(embed_dim)
        self.act = getattr(nn, act)()
        self.head = nn.Linear(embed_dim, pred_dim)

    def _get_compressors(self, same: bool = True) -> (nn.Linear, nn.Linear):
        c = nn.Linear(self.seq_len, int(self.compression_factor * self.seq_len), False)
        if same:
            return c, c
        else:
            return c, nn.Linear(self.seq_len, int(self.compression_factor * self.seq_len), False)

    def configure_optimizer(self,
                            lr: float,
                            weight_decay: float
                            ) -> (torch.optim.Optimizer, torch.optim.Optimizer):
        # decay = set()
        # no_decay = set()

        lrs = [
            lr * (0.95 ** i)
            for i in reversed(
                range(len(self.blocks) + 2)
            )
        ]

        embed = set()
        norm_blocks = [set() for _ in range(len(self.blocks))]
        blocks = [set() for _ in range(len(self.blocks))]
        norm_head = set()
        head = set()

        params = {pn: p for pn, p in self.named_parameters()}
        for pn, p in params.items():
            # print(pn)
            if 'norm' not in pn:
                if 'bias' in pn:
                    nn.init.zeros_(p)
                else:
                    nn.init.kaiming_uniform_(p, a=5**0.5)
                    # nn.init.normal_(p, std=0.02)
                    # nn.init.kaiming_normal_(p, a=5 ** 0.5)
                    # nn.init.xavier_uniform_(p, gain=1 / (2 ** 0.5))

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

            # if any(t in pn for t in ('embed', 'norm', 'bias')):
            #     no_decay.add(pn)
            # else:
            #     decay.add(pn)

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

        # assert len(decay & no_decay) == 0
        # p = params.keys() - (decay | no_decay)
        # assert len(p) == 0, str(p)
        # decay = [params[i] for i in list(decay)]
        # no_decay = [params[i] for i in list(no_decay)]
        # optim_groups = [
        #     {'params': decay, 'weight_decay': weight_decay},
        #     {'params': no_decay, 'weight_decay': 0.0}
        # ]

        optimizer = torch.optim.AdamW(optim_groups)
        return optimizer

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(x, mask)
        for i, block in enumerate(self.blocks):
            x = block(x, self.get_pool(i))
        x = self.norm(x)
        x = self.act(x)
        x = self.head(x)
        if self.out_log:
            x = F.log_softmax(x, -1)
        return x


if __name__ == '__main__':
    from configs.model_cfg import cfg
    opt = Transformer(**cfg).configure_optimizer(0.1, 0.1)
