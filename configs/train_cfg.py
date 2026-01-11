import os

from easydict import EasyDict
from configs.data_cfg import cfg as data_cfg
from configs.model_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()
cfg.seed = 0

cfg.batch_size = 2*1024  # 16
cfg.lr = 6e-3  # 1 3 5
cfg.lr_decay_factor = 1e-3

cfg.weight_decay = 0.001
cfg.target = 'num'  # cat num
cfg.num_masks = 1

cfg.loss = 'MSELoss'  # CrossEntropyLoss SmoothL1Loss KLDivLoss L1Loss MSELoss
cfg.loss_args = {}  # 'reduction': 'batchmean'}
cfg.decay = 'cosine'  # cosine

cfg.accelerator_args = {'mixed_precision': 'fp16', 'cpu': True}

model_cfg.pred_dim = 1  # model_cfg.num_embed_features[0]  # 1  #
cfg.model_cfg = model_cfg
cfg.data_cfg = data_cfg

cfg.exp_dir = os.path.join(ROOT_DIR, 'exp_dir')  # , 'train')
# cfg.lr_decay_by_block = 0.95
cfg.load_pretrained = os.path.join(ROOT_DIR, 'exp_dir', 'MaskedTableModeling.pt')
# 'MaskedTableAutoencoder.pt')  # 'pretrain', "transformer.pt") MaskedTableModeling

cfg.num_epoch = 125
cfg.model = 'PricePrediction'  # PricePrediction TabM PricePredEnsemble


"""
cpu
Epoch 1/125
train loss: 1.2511 - metric: 0.478 - time: 9.6 (9.6) 
valid loss: 0.7318 - metric: 0.426 - time: 9.6 (0.5) 
best
Epoch 2/125
train loss: 0.3328 - metric: 0.245 - time: 19.8 (10.2) 
valid loss: 0.1200 - metric: 0.140 - time: 19.8 (0.6) 
best
Epoch 3/125
train loss: 0.1235 - metric: 0.144 - time: 28.8 (9.0) 
valid loss: 0.0729 - metric: 0.118 - time: 28.8 (0.7) 
best
Epoch 4/125
train loss: 0.0787 - metric: 0.111 - time: 37.8 (9.1) 
valid loss: 0.0579 - metric: 0.098 - time: 37.8 (0.6) 
best
Epoch 5/125
train loss: 0.0655 - metric: 0.101 - time: 46.9 (9.1) 
valid loss: 0.0529 - metric: 0.094 - time: 46.9 (0.7) 
best
Epoch 6/125
train loss: 0.0584 - metric: 0.095 - time: 55.8 (8.9) 
valid loss: 0.0484 - metric: 0.084 - time: 55.8 (0.6) 
best
Epoch 7/125
train loss: 0.0538 - metric: 0.091 - time: 64.7 (8.9) 
valid loss: 0.0518 - metric: 0.088 - time: 64.7 (0.6) 
Epoch 8/125
train loss: 0.0510 - metric: 0.089 - time: 74.2 (9.5) 
valid loss: 0.0538 - metric: 0.092 - time: 74.2 (0.6) 
Epoch 9/125
train loss: 0.0482 - metric: 0.086 - time: 83.2 (8.9) 
valid loss: 0.0432 - metric: 0.081 - time: 83.2 (0.6) 
best
Epoch 10/125
train loss: 0.0456 - metric: 0.084 - time: 92.4 (9.2) 
valid loss: 0.0380 - metric: 0.076 - time: 92.4 (0.7) 
best
Epoch 11/125
train loss: 0.0427 - metric: 0.081 - time: 101.8 (9.4) 
valid loss: 0.0380 - metric: 0.076 - time: 101.8 (0.7) 
Epoch 12/125
train loss: 0.0412 - metric: 0.079 - time: 110.9 (9.1) 
valid loss: 0.0411 - metric: 0.081 - time: 110.9 (0.6) 
Epoch 13/125
train loss: 0.0415 - metric: 0.080 - time: 119.9 (9.0) 
valid loss: 0.0400 - metric: 0.076 - time: 119.9 (0.6) 
Epoch 14/125
train loss: 0.0397 - metric: 0.078 - time: 129.2 (9.2) 
valid loss: 0.0336 - metric: 0.070 - time: 129.2 (0.6) 
best
Epoch 15/125
train loss: 0.0370 - metric: 0.075 - time: 138.2 (9.1) 
valid loss: 0.0364 - metric: 0.077 - time: 138.2 (0.7) 
Epoch 16/125
train loss: 0.0369 - metric: 0.075 - time: 147.6 (9.4) 
valid loss: 0.0341 - metric: 0.070 - time: 147.6 (0.6) 
Epoch 17/125
train loss: 0.0359 - metric: 0.075 - time: 156.8 (9.2) 
valid loss: 0.0371 - metric: 0.072 - time: 156.8 (0.6) 
Epoch 18/125
train loss: 0.0353 - metric: 0.074 - time: 165.7 (9.0) 
valid loss: 0.0334 - metric: 0.071 - time: 165.7 (0.6) 
Epoch 19/125
train loss: 0.0343 - metric: 0.073 - time: 174.8 (9.1) 
valid loss: 0.0308 - metric: 0.066 - time: 174.8 (0.6) 
best
Epoch 20/125
train loss: 0.0325 - metric: 0.070 - time: 184.3 (9.5) 
valid loss: 0.0313 - metric: 0.066 - time: 184.3 (0.6) 
best
Epoch 21/125
train loss: 0.0322 - metric: 0.071 - time: 193.5 (9.2) 
valid loss: 0.0331 - metric: 0.068 - time: 193.5 (0.6) 
Epoch 22/125
train loss: 0.0306 - metric: 0.069 - time: 202.6 (9.0) 
valid loss: 0.0324 - metric: 0.067 - time: 202.6 (0.6) 
Epoch 23/125
train loss: 0.0307 - metric: 0.069 - time: 211.9 (9.3) 
valid loss: 0.0307 - metric: 0.065 - time: 211.9 (0.6) 
best
Epoch 24/125
train loss: 0.0312 - metric: 0.070 - time: 221.1 (9.2) 
valid loss: 0.0302 - metric: 0.064 - time: 221.1 (0.7) 
best
Epoch 25/125
train loss: 0.0307 - metric: 0.069 - time: 230.2 (9.1) 
valid loss: 0.0309 - metric: 0.065 - time: 230.2 (0.6) 
Epoch 26/125
train loss: 0.0296 - metric: 0.068 - time: 239.6 (9.4) 
valid loss: 0.0319 - metric: 0.068 - time: 239.6 (0.7) 
Epoch 27/125
train loss: 0.0277 - metric: 0.066 - time: 248.6 (9.0) 
valid loss: 0.0301 - metric: 0.065 - time: 248.6 (0.6) 
Epoch 28/125
train loss: 0.0280 - metric: 0.066 - time: 257.6 (9.1) 
valid loss: 0.0327 - metric: 0.067 - time: 257.6 (0.7) 
Epoch 29/125
train loss: 0.0284 - metric: 0.067 - time: 266.6 (8.9) 
valid loss: 0.0305 - metric: 0.063 - time: 266.6 (0.7) 
best
Epoch 30/125
train loss: 0.0270 - metric: 0.065 - time: 275.7 (9.1) 
valid loss: 0.0285 - metric: 0.062 - time: 275.7 (0.6) 
best



"""