import os

from easydict import EasyDict
from configs.data_cfg import cfg as data_cfg
from configs.model_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()
cfg.seed = 0
cfg.exp_dir = os.path.join(ROOT_DIR, 'exp_dir')

cfg.batch_size = 8 * 1024
cfg.num_epoch = 125

cfg.lr = 1e-2
cfg.lr_decay_factor = 1e-2
cfg.lr_decay = 'cosine'

model_cfg.log_softmax = True  # False True
cfg.loss = 'KLDivLoss'  # CrossEntropyLoss SmoothL1Loss KLDivLoss L1Loss
cfg.loss_args = {'reduction': 'batchmean'}
cfg.weight_decay = 1e-5

cfg.accelerator_args = {'mixed_precision': 'fp16', 'cpu': True}
cfg.target_type = 'mask'

cfg.model = 'MaskedTableAutoencoder'
cfg.mask_ratio = 0.65
model_cfg.decoder_embed_dim = model_cfg.embed_dim // 2
model_cfg.decoder_num_heads = model_cfg.num_heads
model_cfg.decoder_num_blocks = max(1, model_cfg.num_blocks // 3)

# cfg.model = 'MaskedTableModeling'
# cfg.mask_ratio = 0.5

model_cfg.pred_dim = sum(model_cfg.num_embed_features)

data_cfg.target_type = cfg.target_type
data_cfg.mask_first_token = len(data_cfg.features) == len(model_cfg.num_embed_features)

cfg.data_cfg = data_cfg
cfg.model_cfg = model_cfg

"""
cpu
Epoch 1/125
train loss: 6.7926 - metric: 0.102 - time: 9.3 (9.3) 
valid loss: 6.5399 - metric: 0.370 - time: 9.3 (0.7) 
best
Epoch 2/125
train loss: 6.2366 - metric: 0.384 - time: 18.5 (9.2) 
valid loss: 5.7964 - metric: 0.386 - time: 18.5 (0.8) 
best
Epoch 3/125
train loss: 5.2178 - metric: 0.394 - time: 27.9 (9.3) 
valid loss: 4.5604 - metric: 0.395 - time: 27.9 (0.8) 
best
Epoch 4/125
train loss: 3.8003 - metric: 0.398 - time: 37.1 (9.2) 
valid loss: 3.0929 - metric: 0.398 - time: 37.1 (0.7) 
best
Epoch 5/125
train loss: 2.7200 - metric: 0.404 - time: 46.4 (9.3) 
valid loss: 2.4100 - metric: 0.402 - time: 46.4 (0.7) 
best
Epoch 6/125
train loss: 2.3492 - metric: 0.410 - time: 55.6 (9.2) 
valid loss: 2.2262 - metric: 0.417 - time: 55.6 (0.9) 
best
Epoch 7/125
train loss: 2.2124 - metric: 0.427 - time: 65.3 (9.7) 
valid loss: 2.0946 - metric: 0.438 - time: 65.3 (0.8) 
best
Epoch 8/125
train loss: 2.1037 - metric: 0.448 - time: 75.3 (10.0) 
valid loss: 2.0899 - metric: 0.464 - time: 75.3 (0.8) 
best
Epoch 9/125
train loss: 2.0076 - metric: 0.474 - time: 84.5 (9.2) 
valid loss: 1.8794 - metric: 0.488 - time: 84.5 (0.8) 
best
Epoch 10/125
train loss: 1.9268 - metric: 0.494 - time: 94.0 (9.5) 
valid loss: 1.8792 - metric: 0.507 - time: 94.0 (0.7) 
best
Epoch 11/125
"""
