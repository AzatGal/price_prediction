from easydict import EasyDict

from configs.data_cfg import cfg as data_cfg

cfg = EasyDict()

cfg.num_q_heads = 4
cfg.num_kv_heads = 2
cfg.embed_dim = 4 * cfg.num_q_heads  # 24
cfg.num_blocks = 24
cfg.act = 'SiLU'  # SiLU

cfg.pool = 'mean'  # mean token
# cfg.include_target = False  # False True
cfg.mask_first_token = False
cfg.add_first_token = cfg.pool == 'token' and not cfg.mask_first_token
cfg.num_embed_features = (
    data_cfg.data_transformer.num_bins[0 if cfg.mask_first_token else 1:] +
    data_cfg.data_transformer.num_cats
)

cfg.attn_dropout = 0.05
cfg.mlp_dropout = 0.1
cfg.dropout = 0.1
cfg.kv_compression = 'KV'  # Head KV Layer
cfg.kv_compression_ratio = 0.01
cfg.mlp_dim_factor = 1  # 3 / 2

cfg.attn = 'Attention'  # Linear
cfg.mlp = 'GLUMLP'
cfg.norm = 'LayerNorm'

