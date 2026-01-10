import os

from easydict import EasyDict
from configs.data_cfg import cfg as data_cfg
from configs.model_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()
cfg.seed = 0

cfg.batch_size = 64  # 16
cfg.lr = 6e-3  # 1 3 5
cfg.lr_decay_factor = 1e-3
# cfg.lr_decay_by_block = 0.5

cfg.weight_decay = 0.00001
cfg.target = 'num'  # cat num
cfg.num_masks = 1

cfg.loss = 'MSELoss'  # CrossEntropyLoss SmoothL1Loss KLDivLoss L1Loss MSELoss
cfg.loss_args = {}  # 'reduction': 'batchmean'}
cfg.decay = 'linear'  # cosine

cfg.accelerator_args = {'mixed_precision': 'fp16', 'cpu': True}

model_cfg.pred_dim = 1  # model_cfg.num_embed_features[0]  # 1  #
cfg.model_cfg = model_cfg
cfg.data_cfg = data_cfg

cfg.exp_dir = os.path.join(ROOT_DIR, 'exp_dir')  # , 'train')
# cfg.load_pretrained = os.path.join(ROOT_DIR, 'exp_dir', 'MaskedTableModeling.pt')
# 'MaskedTableAutoencoder.pt')  # 'pretrain', "transformer.pt") MaskedTableModeling

cfg.num_epoch = 125
cfg.model = 'TabM'  # PricePrediction TabM PricePredEnsemble


"""
cpu
Epoch 1/125
train loss: 0.5971 - metric: 0.336 - time: 6.2 (6.2) 
valid loss: 0.1826 - metric: 0.178 - time: 6.2 (0.3) 
best
Epoch 2/125
train loss: 0.1493 - metric: 0.155 - time: 12.2 (6.0) 
valid loss: 0.0783 - metric: 0.103 - time: 12.2 (0.3) 
best
Epoch 3/125
train loss: 0.0813 - metric: 0.114 - time: 18.2 (6.0) 
valid loss: 0.0513 - metric: 0.086 - time: 18.2 (0.3) 
best
Epoch 4/125
train loss: 0.0604 - metric: 0.097 - time: 24.3 (6.1) 
valid loss: 0.0534 - metric: 0.088 - time: 24.3 (0.3) 
Epoch 5/125
train loss: 0.0525 - metric: 0.091 - time: 30.5 (6.2) 
valid loss: 0.0505 - metric: 0.092 - time: 30.5 (0.3) 
Epoch 6/125
train loss: 0.0462 - metric: 0.084 - time: 36.5 (6.1) 
valid loss: 0.0382 - metric: 0.076 - time: 36.5 (0.3) 
best
Epoch 7/125
train loss: 0.0434 - metric: 0.082 - time: 42.9 (6.4) 
valid loss: 0.0461 - metric: 0.087 - time: 42.9 (0.3) 
Epoch 8/125
train loss: 0.0389 - metric: 0.077 - time: 49.1 (6.2) 
valid loss: 0.0352 - metric: 0.071 - time: 49.1 (0.3) 
best
Epoch 9/125
train loss: 0.0363 - metric: 0.075 - time: 55.6 (6.5) 
valid loss: 0.0340 - metric: 0.070 - time: 55.6 (0.3) 
best
Epoch 10/125
train loss: 0.0344 - metric: 0.073 - time: 61.7 (6.2) 
valid loss: 0.0378 - metric: 0.081 - time: 61.7 (0.3) 
Epoch 11/125
train loss: 0.0335 - metric: 0.072 - time: 67.9 (6.1) 
valid loss: 0.0332 - metric: 0.068 - time: 67.9 (0.3) 
best
Epoch 12/125
train loss: 0.0300 - metric: 0.069 - time: 74.0 (6.2) 
valid loss: 0.0314 - metric: 0.067 - time: 74.0 (0.3) 
best
Epoch 13/125
train loss: 0.0279 - metric: 0.066 - time: 80.4 (6.3) 
valid loss: 0.0312 - metric: 0.066 - time: 80.4 (0.3) 
best
Epoch 14/125
train loss: 0.0277 - metric: 0.066 - time: 86.8 (6.4) 
valid loss: 0.0354 - metric: 0.068 - time: 86.8 (0.3) 
Epoch 15/125
train loss: 0.0265 - metric: 0.065 - time: 92.9 (6.1) 
valid loss: 0.0297 - metric: 0.064 - time: 92.9 (0.3) 
best
Epoch 16/125
train loss: 0.0273 - metric: 0.066 - time: 99.1 (6.2) 
valid loss: 0.0315 - metric: 0.071 - time: 99.1 (0.3) 
Epoch 17/125
train loss: 0.0280 - metric: 0.068 - time: 105.2 (6.1) 
valid loss: 0.0286 - metric: 0.063 - time: 105.2 (0.3) 
best
Epoch 18/125
train loss: 0.0250 - metric: 0.063 - time: 111.4 (6.3) 
valid loss: 0.0330 - metric: 0.066 - time: 111.4 (0.3) 
Epoch 19/125
train loss: 0.0246 - metric: 0.063 - time: 117.5 (6.1) 
valid loss: 0.0299 - metric: 0.064 - time: 117.5 (0.3) 
Epoch 20/125
train loss: 0.0224 - metric: 0.061 - time: 123.6 (6.1) 
valid loss: 0.0267 - metric: 0.059 - time: 123.6 (0.3) 
best
Epoch 21/125
train loss: 0.0218 - metric: 0.060 - time: 129.9 (6.3) 
valid loss: 0.0300 - metric: 0.063 - time: 129.9 (0.3) 
Epoch 22/125
"""