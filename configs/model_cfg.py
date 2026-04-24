from easydict import EasyDict

from configs.data_cfg import cfg as data_cfg

cfg = EasyDict()

cfg.num_q_heads = 12
cfg.num_kv_heads = 6  # 4
cfg.embed_dim = 8 * cfg.num_q_heads  # 8 10 * cfg.num_q_heads
cfg.num_blocks = 34  # 10 24 40
cfg.act = 'ReLU'  # SiLU

cfg.pool = 'cls'  # avg cls sum w_avg
cfg.mask_first_token = False  # True False
cfg.add_cls_token = cfg.pool == 'cls' and not cfg.mask_first_token
cfg.num_embed_features = (
    data_cfg.data_transformer.num_bins[int(not cfg.mask_first_token):] +
    data_cfg.data_transformer.num_cats
)

cfg.attn_dropout = 0.1  # 1
cfg.mlp_dropout = 0.1
cfg.dropout = 0.1
cfg.kv_compression = 'Head'  # Head KV Layer   # None
cfg.kv_compression_dim = 1  # 0.01
cfg.mlp_dim_factor = 2  # 2 8 / 3

cfg.attn = 'Attention'  # Linear
cfg.mlp = 'GatedMLP'  # GatedMLP MLP
cfg.norm = 'RMSNorm'  # LayerNorm RMSNorm


# cfg.embed_dim = 160  # 192  # 160  # 8 * cfg.num_q_heads  # 8 10 * cfg.num_q_heads
# cfg.num_blocks = 40  # 10 24 40
# cfg.act = 'ReLU'  # SiLU
#
# cfg.mask_first_token = False  # True False
# cfg.num_embed_features = (
#     data_cfg.data_transformer.num_bins[int(not cfg.mask_first_token):] +
#     data_cfg.data_transformer.num_cats
# )
#
# # cfg.comp_dropout = 0.1
# cfg.mlp_dropout = 0.1
# cfg.dropout = 0.1
# cfg.mlp_dim_factor = 2  # 2 8 / 3
# # cfg.comp_dim_factor = 2  # 2 8 / 3
#
# cfg.compressor = 'Compressor'
# cfg.mlp = 'GatedMLP'  # GatedMLP MLP
# cfg.norm = 'RMSNorm'  # LayerNorm RMSNorm


