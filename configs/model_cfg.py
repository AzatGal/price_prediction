from easydict import EasyDict

from configs.data_cfg import cfg as data_cfg

cfg = EasyDict()

cfg.num_q_heads = 12  # 16
cfg.num_kv_heads = 4  # 8
cfg.embed_dim = 64 * cfg.num_q_heads  # 8 10 * cfg.num_q_heads
cfg.num_blocks = 12  # 10  # 40
cfg.act = 'ReLU'  # SiLU

cfg.pool = 'avg'  # avg cls sum
cfg.mask_first_token = False  # True False
cfg.add_first_token = cfg.pool == 'cls' and not cfg.mask_first_token
cfg.num_embed_features = (
    data_cfg.data_transformer.num_bins[int(not cfg.mask_first_token):] +
    data_cfg.data_transformer.num_cats
)

cfg.attn_dropout = 0.0
cfg.mlp_dropout = 0.1
cfg.dropout = 0.1
cfg.kv_compression = 'Head'  # Head KV Layer
cfg.kv_compression_ratio = 0.15  # 0.01
cfg.mlp_dim_factor = 2  # 2 8 / 3

cfg.attn = 'Attention'  # Linear
cfg.mlp = 'GatedMLP'  # GatedMLP MLP
cfg.norm = 'RMSNorm'  # LayerNorm RMSNorm

