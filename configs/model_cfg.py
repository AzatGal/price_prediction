from easydict import EasyDict

# from configs.data_cfg import cfg as data_cfg

cfg = EasyDict()

# cfg.k = 2
cfg.k = 8  # 16
cfg.embed_dim = 8  # 32
cfg.num_blocks = 12  # 3
cfg.act = 'ReLU'  # SiLU
cfg.pred_dim = 1

cfg.pool = 'cls'  # avg cls sum w_avg
cfg.add_cls_token = cfg.pool == 'cls'

cfg.attn_dropout = 0.0  # 1
cfg.mlp_dropout = 0.0
cfg.dropout = 0.0

cfg.kv_compression = 'Head'  # Head KV Layer  None
cfg.kv_compression_dim = 8  # 0.01
cfg.mlp_dim_factor = 3 / 2  # 2 8 / 3

cfg.attn = 'AttentionEnsemble'  # Linear
cfg.mlp = 'GatedMLP'  # GatedMLP MLP
cfg.norm = 'RMSNorm'  # LayerNorm RMSNorm



