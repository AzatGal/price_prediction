from easydict import EasyDict

from configs.data_cfg import cfg as data_cfg

cfg = EasyDict()

cfg.num_heads = 3
cfg.embed_dim = 16 * cfg.num_heads
cfg.num_blocks = 2
cfg.act = 'SiLU'
cfg.num_embed_features = (data_cfg.data_transformer.num_bins +
                          data_cfg.data_transformer.num_categories)
cfg.pred_dim = 1  # cfg.num_embed_features[0]
cfg.attn_dropout = 0.0
cfg.mlp_dropout = 0.1
cfg.dropout = 0.1
cfg.compression_factor = 0.5
cfg.compression = 'Head'  # Head KV Layer
cfg.mlp_dim_factor = 2  # 5 / 3
cfg.attn = 'Attn'  # Linear
cfg.mlp = 'GLUMLP'
cfg.norm = 'LayerNorm'
cfg.out_log = False

"""
Epoch 1/126
train loss: 0.7802 - metric: 0.476 - time: 6.2 
valid loss: 0.7886 - metric: 0.500 - time: 0.3 
best
Epoch 2/126
train loss: 0.7632 - metric: 0.486 - time: 6.0 
valid loss: 0.7678 - metric: 0.493 - time: 0.3 
best
Epoch 3/126
train loss: 0.7377 - metric: 0.467 - time: 6.2 
valid loss: 0.7202 - metric: 0.463 - time: 0.3 
best
Epoch 4/126
train loss: 0.6782 - metric: 0.419 - time: 6.3 
valid loss: 0.6006 - metric: 0.355 - time: 0.3 
best
Epoch 5/126
train loss: 0.5765 - metric: 0.334 - time: 6.3 
valid loss: 0.4780 - metric: 0.265 - time: 0.3 
best
Epoch 6/126
train loss: 0.4915 - metric: 0.271 - time: 6.2 
valid loss: 0.4112 - metric: 0.219 - time: 0.3 
best
Epoch 7/126
train loss: 0.4374 - metric: 0.235 - time: 6.3 
valid loss: 0.3817 - metric: 0.203 - time: 0.3 
best
Epoch 8/126
train loss: 0.3996 - metric: 0.215 - time: 6.2 
valid loss: 0.3529 - metric: 0.191 - time: 0.3 
best
Epoch 9/126
train loss: 0.3730 - metric: 0.201 - time: 6.2 
valid loss: 0.3336 - metric: 0.178 - time: 0.3 
best
Epoch 10/126
train loss: 0.3528 - metric: 0.191 - time: 6.1 
valid loss: 0.3135 - metric: 0.168 - time: 0.3 
best
Epoch 11/126
train loss: 0.3335 - metric: 0.180 - time: 6.1 
valid loss: 0.3036 - metric: 0.162 - time: 0.3 
best
Epoch 12/126
train loss: 0.3206 - metric: 0.174 - time: 6.1 
valid loss: 0.2918 - metric: 0.155 - time: 0.3 
best
Epoch 13/126
train loss: 0.3107 - metric: 0.168 - time: 6.1 
valid loss: 0.2899 - metric: 0.156 - time: 0.3 
Epoch 14/126
train loss: 0.3013 - metric: 0.163 - time: 6.1 
valid loss: 0.2824 - metric: 0.149 - time: 0.3 
best
Epoch 15/126
train loss: 0.2917 - metric: 0.158 - time: 6.1 
valid loss: 0.2751 - metric: 0.146 - time: 0.3 
best
Epoch 16/126
train loss: 0.2876 - metric: 0.156 - time: 6.1 
valid loss: 0.2653 - metric: 0.141 - time: 0.3 
best
Epoch 17/126
train loss: 0.2791 - metric: 0.152 - time: 6.1 
valid loss: 0.2631 - metric: 0.139 - time: 0.4 
best
Epoch 18/126
train loss: 0.2738 - metric: 0.149 - time: 6.1 
valid loss: 0.2598 - metric: 0.138 - time: 0.3 
best
Epoch 19/126
train loss: 0.2673 - metric: 0.145 - time: 6.2 
valid loss: 0.2536 - metric: 0.135 - time: 0.3 
best
Epoch 20/126
train loss: 0.2644 - metric: 0.144 - time: 6.2 
valid loss: 0.2494 - metric: 0.133 - time: 0.3 
best
Epoch 21/126

"""
