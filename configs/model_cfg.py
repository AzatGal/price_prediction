from easydict import EasyDict


cfg = EasyDict()

# cfg.k = 2
# cfg.embed_dim = 4
cfg.k = 64 # 32 # 8
cfg.embed_dim = 8 # 128  # 32
cfg.num_blocks = 2 # 3
cfg.act = 'ReLU'  # SiLU
cfg.pred_dim = 1  # 2  # 1

# cfg.pool = 'cls'  # avg cls sum w_avg
# cfg.add_cls_token = cfg.pool == 'cls'

cfg.attn_dropout = 0.0  # 1
cfg.mlp_dropout = 0.1
cfg.dropout = 0.1

cfg.kv_compression_ratio = 0.2  # 2
cfg.mlp_dim_factor = 3 / 2  # 2 8 / 3

cfg.attn_bias = False  # True
cfg.mlp_bias = False  # True

cfg.share_weights = True

# cfg.model = 'TransformerEnsemble'

