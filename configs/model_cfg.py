from easydict import EasyDict

from configs.data_cfg import cfg as data_cfg

cfg = EasyDict()

cfg.num_heads = 4
cfg.embed_dim = 24 * cfg.num_heads
cfg.num_blocks = 3
cfg.act = 'SiLU'  # SiLU
cfg.num_embed_features = (data_cfg.data_transformer.num_bins +
                          data_cfg.data_transformer.num_cats)
# cfg.pred_dim = 1  # cfg.num_embed_features[0]  # 1  #
cfg.attn_dropout = 0.0
cfg.mlp_dropout = 0.1
cfg.dropout = 0.1
cfg.compression_factor = 0.25
cfg.compression = 'Head'  # Head KV Layer
cfg.mlp_dim_factor = 1  # 5 / 3
cfg.attn = 'Attn'  # Linear
cfg.mlp = 'GLUMLP'
cfg.norm = 'LayerNorm'
cfg.log_softmax = False  # False True


"""
Epoch 1/128
train loss: 0.8105 - metric: 0.575 - time: 8.9 
valid loss: 0.7552 - metric: 0.490 - time: 0.4 
best
Epoch 2/128
train loss: 0.7561 - metric: 0.479 - time: 8.2 
valid loss: 0.6962 - metric: 0.432 - time: 0.4 
best
Epoch 3/128
train loss: 0.6260 - metric: 0.370 - time: 8.2 
valid loss: 0.4860 - metric: 0.262 - time: 0.4 
best
Epoch 4/128
train loss: 0.4739 - metric: 0.260 - time: 8.3 
valid loss: 0.4076 - metric: 0.214 - time: 0.3 
best
Epoch 5/128
train loss: 0.4110 - metric: 0.224 - time: 8.0 
valid loss: 0.3533 - metric: 0.187 - time: 0.3 
best
Epoch 6/128
train loss: 0.3658 - metric: 0.199 - time: 8.7 
valid loss: 0.3176 - metric: 0.169 - time: 0.4 
best
Epoch 7/128
train loss: 0.3344 - metric: 0.182 - time: 8.0 
valid loss: 0.2886 - metric: 0.153 - time: 0.3 
best
Epoch 8/128
train loss: 0.3099 - metric: 0.169 - time: 8.7 
valid loss: 0.2731 - metric: 0.145 - time: 0.3 
best
Epoch 9/128
train loss: 0.2930 - metric: 0.159 - time: 8.2 
valid loss: 0.2552 - metric: 0.135 - time: 0.4 
best
Epoch 10/128
train loss: 0.2777 - metric: 0.151 - time: 8.3 
valid loss: 0.2436 - metric: 0.133 - time: 0.4 
best
Epoch 11/128


"""
