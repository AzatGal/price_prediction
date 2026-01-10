import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embedding import FeatureEmbedding


class FeatureEmbeddingEnsemble(nn.Module):
    def __init__(self,
                 k: int,
                 num_embed_features: list[int],
                 embed_dim: int,
                 norm: str,
                 dropout: float
                 ) -> None:
        super().__init__()
        # self.num_embed_features = torch.tensor(num_embed_features)
        self.k = k
        self.embed_dim = embed_dim
        self.num_embeds = sum(num_embed_features) + 1
        self.seq_len = len(num_embed_features)
        # self.mask_token = nn.Parameter(torch)
        self.weight = nn.Parameter(torch.empty(self.num_embeds * k, embed_dim))
        self.pos_embed = nn.Parameter(torch.empty(k, self.seq_len, embed_dim))
        self.register_buffer('offsets',
                             torch.arange(k).unsqueeze(1) * self.num_embeds)
        # self.norm = getattr(nn, norm)(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x.unsqueeze(1).repeat(1, self.k, 1)
        x = x + self.offsets
        if mask is not None:
            # print(x.shape)
            # print(x[mask.unsqueeze(1).repeat(1, self.k, 1)].shape)
            for i in range(self.k):
                x[:, i] = x[:, i].masked_fill(mask, (self.offsets[i] + self.num_embeds - 1).item())
            # x[mask.unsqueeze(1).repeat(1, self.k, 1)].reshape(x.size(0), self.k, -1) = self.offsets - 1
            # x = x.masked_fill(mask, self.num_embeds - 1)
            # i = torch.randint(0, mask.size(1), (1,)).item()
            # x[torch.arange(x.size(0)), i] = self.num_embeds
        # print(x.max())
        x = F.embedding(x, self.weight)
        # x = x.reshape(-1, self.k, self.seq_len, self.embed_dim)
        x = x + self.pos_embed
        # x = self.norm(x)
        x = self.dropout(x)
        return x


class LinearEnsemble(nn.Module):
    def __init__(self,
                 k: int,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(k, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(k, 1, out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, k, T, C]
        # print('lex', x.shape)
        x = x @ self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class AttnEnsemble(nn.Module):
    def __init__(self,
                 k: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 k_compressor: LinearEnsemble = None,
                 v_compressor: LinearEnsemble = None,
                 ) -> None:
        super().__init__()
        self.dropout = dropout
        self.k = k
        self.embed_dim = embed_dim
        self.qkv_proj = LinearEnsemble(k, embed_dim, 3 * embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.out_proj = LinearEnsemble(k, embed_dim, embed_dim)

        self.k_compressor = k_compressor
        self.v_compressor = v_compressor
        if k_compressor is not None:
            self.seq_len = self.k_compressor.in_features
            self.register_buffer('ids', torch.arange(self.seq_len).reshape(1, 1, 1, self.seq_len, 1))

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor = None,
                mask_only_attn: bool = False
                ) -> torch.Tensor:
        # x: [B, k, T, C]
        B, _, T, _ = x.shape
        q, k, v = (self.qkv_proj(x)
                   .reshape(B, self.k, T, self.num_heads, 3, self.head_dim)
                   .permute(0, 3, 1, 2, 4, 5)
                   .unbind(4))
        # print('q', q.shape)
        if mask is not None:
            # print('mask', mask.shape)
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        if mask_only_attn:
            q = q[mask].reshape(B, self.num_heads, self.k, -1, self.head_dim)
        if self.k_compressor is not None:
            # if T < self.seq_len:
            #     t = torch.zeros(
            #         B, self.k, self.num_heads, self.seq_len, self.embed_dim,
            #         dtype=k.dtype, device=k.device
            #     ).scatter_(
            #         2,
            #         (self.ids
            #          .expand(B, self.k, self.num_heads, -1, self.embed_dim)[mask]
            #          .reshape(B, self.k, self.num_heads, -1, self.embed_dim)),
            #         k
            #     )
            #     k = t
            k = self.k_compressor(
                k.transpose(3, 4)
            ).transpose(3, 4)
        if self.v_compressor is not None:
            # if T < self.seq_len:
            #     t = torch.zeros(
            #         B, self.k, self.num_heads, self.seq_len, self.embed_dim,
            #         dtype=k.dtype, device=k.device
            #     ).scatter_(
            #         2,
            #         (self.ids
            #          .expand(B, self.k, self.num_heads, -1, self.embed_dim)[mask]
            #          .reshape(B, self.k, self.num_heads, -1, self.embed_dim)),
            #         v
            #     )
            #     v = t
            v = self.v_compressor(
                v.transpose(3, 4)
            ).transpose(3, 4)
        a = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0
        )
        # print('a', a.shape)
        a = self.out_proj(
            a
            .transpose(2, 3)
            .reshape(B, self.k, -1, self.embed_dim)
        )
        # print(a.shape)
        return a


class GLUMLPEnsemble(nn.Module):
    def __init__(self,
                 k: int,
                 embed_dim: int,
                 dim_factor: float,
                 dropout: float,
                 act: str,
                 bias: bool = False
                 ) -> None:
        super().__init__()
        self.in_proj = LinearEnsemble(k, embed_dim, 2 * int(dim_factor * embed_dim), bias)
        self.out_proj = LinearEnsemble(k, int(dim_factor * embed_dim), embed_dim, bias)
        self.dropout = nn.Dropout(dropout)
        self.act = getattr(nn, act)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = self.in_proj(x).chunk(2, dim=-1)
        x = self.act(x) * y
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BlockEnsemble(nn.Module):
    def __init__(self,
                 k: int,
                 embed_dim: int,
                 num_heads: int,
                 attn_dropout: float,
                 mlp_dropout: float,
                 dropout: float,
                 act: str,
                 mlp_dim_factor: float,
                 norm: str,
                 k_compressor: LinearEnsemble = None,
                 v_compressor: LinearEnsemble = None
                 ) -> None:
        super().__init__()
        self.attn_norm = getattr(nn, norm)(embed_dim, elementwise_affine=False)
        self.mlp_norm = getattr(nn, norm)(embed_dim, elementwise_affine=False)

        self.attn_drop = nn.Dropout(dropout)
        self.mlp_drop = nn.Dropout(dropout)

        self.mlp = GLUMLPEnsemble(k, embed_dim, mlp_dim_factor, mlp_dropout, act)
        self.attn = AttnEnsemble(k, embed_dim, num_heads, attn_dropout, k_compressor, v_compressor)

    def _attn_block(self,
                    x: torch.Tensor,
                    mask: torch.Tensor,
                    norm_attn: bool,
                    mask_only_attn: bool
                    ) -> torch.Tensor:
        if norm_attn:
            y = self.attn_norm(x)
        else:
            y = x
        # y = self.attn_norm(x)
        y = self.attn(y, mask, mask_only_attn)  # y
        y = self.attn_drop(y)
        if mask_only_attn:
            y = y + x[mask].reshape(y.shape)
        else:
            y = y + x
        return y

    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp_norm(x)
        y = self.mlp(y)  # y
        y = self.mlp_drop(y)
        y = y + x
        return y

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor = None,
                norm_attn: bool = True,
                mask_only_attn: bool = False,
                ) -> torch.Tensor:
        x = self._attn_block(x, mask, norm_attn, mask_only_attn)
        x = self._mlp_block(x)
        return x


class PricePredEnsemble(nn.Module):
    def __init__(self,
                 k: int,
                 embed_dim: int,
                 num_heads: int,
                 num_embed_features: list[int],
                 attn_dropout: float,
                 mlp_dropout: float,
                 dropout: float,
                 act: str,
                 mlp_dim_factor: float,
                 num_blocks: int,
                 norm: str,
                 pred_dim: int,
                 log_softmax: bool = False,
                 compression_factor: float = None,
                 compression: str = None,
                 ) -> None:
        super().__init__()
        self.k = k
        self.embed = FeatureEmbeddingEnsemble(k, num_embed_features, embed_dim, norm, dropout)

        self.log_softmax = log_softmax
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

        self.blocks = nn.ModuleList([
            BlockEnsemble(k, embed_dim, num_heads, attn_dropout, mlp_dropout, dropout,
                          act, mlp_dim_factor, norm, *compressors[i])
            for i in range(num_blocks)
        ])
        self.norm = getattr(nn, norm)(embed_dim)
        self.head = LinearEnsemble(k, embed_dim, pred_dim, True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(x, mask)
        # x = x.unsqueeze(1).expand(-1, self.k, -1, -1)
        # print(x.shape)
        # print(mask.shape)
        # mask = mask.unsqueeze(1)
        # print(mask.shape)
        mask = mask.unsqueeze(1).expand(-1, self.k, -1)
        for i, block in enumerate(self.blocks):
            x = block(x, mask, True, i + 1 == len(self.blocks))
        # x = self.norm(x)
        # print(x.shape)
        x = self.head(x)
        # print(x.shape)
        return x.squeeze(2)

    def _get_compressors(self, same: bool = True) -> (LinearEnsemble, LinearEnsemble):
        c = LinearEnsemble(self.k, self.seq_len, int(self.compression_factor * self.seq_len))
        if same:
            return c, c
        else:
            return c, LinearEnsemble(self.k, self.seq_len, int(self.compression_factor * self.seq_len))

    def configure_optimizer(self,
                            lr: float,
                            weight_decay: float,
                            lr_decay_by_block: float = None
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
                    nn.init.kaiming_uniform_(p, a=5**0.5)
                    # nn.init.normal_(p, std=0.02)
                    # nn.init.kaiming_normal_(p, a=5 ** 0.5)
                    # nn.init.xavier_uniform_(p, gain=1 / (2 ** 0.5))

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

        # if self.embed.weight[:-1].shape == self.pred_head.weight.shape:
        #     self.embed.weight[:-1] = self.pred_head.weight

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
            return torch.optim.AdamW(optim_groups, lr=lr)

