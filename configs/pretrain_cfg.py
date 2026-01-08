import os

from easydict import EasyDict
from configs.data_cfg import cfg as data_cfg
from configs.model_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()
cfg.seed = 0

cfg.batch_size = 128
cfg.lr = 3e-3  # 1 3 5
cfg.lr_decay_factor = 1e-3

cfg.weight_decay = 1e-5
cfg.target = 'mask'

cfg.loss = 'CrossEntropyLoss'  # CrossEntropyLoss SmoothL1Loss KLDivLoss L1Loss
cfg.loss_args = {}  # 'reduction': 'batchmean'}
cfg.decay = 'linear'

cfg.accelerator_args = {'mixed_precision': 'fp16', 'cpu': True}

cfg.data_cfg = data_cfg

cfg.exp_dir = os.path.join(ROOT_DIR, 'exp_dir')  # , 'pretrain')
cfg.num_epoch = 125

# cfg.model = 'MaskedTableAutoencoder'
# cfg.num_masks = int(0.75 * len(data_cfg.features))
# model_cfg.decoder_embed_dim = model_cfg.embed_dim // 2
# model_cfg.decoder_num_heads = model_cfg.num_heads
# model_cfg.decoder_num_blocks = max(1, model_cfg.num_blocks // 3)

model_cfg.pred_dim = sum(model_cfg.num_embed_features)  # 1  #
cfg.model_cfg = model_cfg

cfg.model = 'MaskedTableModeling'
cfg.num_masks = int(0.75 * len(data_cfg.features))


"""
mtm 0.75
cpu
Epoch 1/125
train loss: 6.1542 - metric: 0.143 - time: 10.5 (10.5) 
valid loss: 4.3046 - metric: 0.393 - time: 10.5 (0.9) 
best
Epoch 2/125
train loss: 3.1004 - metric: 0.399 - time: 21.0 (10.4) 
valid loss: 2.4152 - metric: 0.408 - time: 21.0 (0.9) 
best
Epoch 3/125
train loss: 2.3393 - metric: 0.412 - time: 35.0 (14.0) 
valid loss: 2.2375 - metric: 0.426 - time: 35.0 (0.9) 
best
Epoch 4/125
train loss: 2.1980 - metric: 0.438 - time: 45.3 (10.4) 
valid loss: 2.1010 - metric: 0.458 - time: 45.3 (0.9) 
best
Epoch 5/125
train loss: 2.0812 - metric: 0.460 - time: 55.8 (10.5) 
valid loss: 1.9880 - metric: 0.474 - time: 55.8 (0.9) 
best
Epoch 6/125
train loss: 2.0073 - metric: 0.474 - time: 66.6 (10.7) 
valid loss: 1.9224 - metric: 0.490 - time: 66.6 (0.9) 
best
Epoch 7/125
train loss: 1.9608 - metric: 0.481 - time: 77.1 (10.5) 
valid loss: 1.8897 - metric: 0.492 - time: 77.1 (1.2) 
best
Epoch 8/125
train loss: 1.9218 - metric: 0.488 - time: 89.1 (12.0) 
valid loss: 1.8408 - metric: 0.504 - time: 89.1 (1.6) 
best
Epoch 9/125
train loss: 1.8944 - metric: 0.493 - time: 100.4 (11.3) 
valid loss: 1.8305 - metric: 0.508 - time: 100.4 (0.9) 
best
Epoch 10/125
train loss: 1.8728 - metric: 0.498 - time: 111.4 (11.1) 
valid loss: 1.7927 - metric: 0.510 - time: 111.4 (0.9) 
best
Epoch 11/125
train loss: 1.8556 - metric: 0.502 - time: 121.8 (10.3) 
valid loss: 1.7913 - metric: 0.518 - time: 121.8 (0.9) 
best
Epoch 12/125
train loss: 1.8463 - metric: 0.503 - time: 131.9 (10.1) 
valid loss: 1.7760 - metric: 0.518 - time: 131.9 (0.9) 
best
Epoch 13/125
train loss: 1.8373 - metric: 0.505 - time: 142.0 (10.1) 
valid loss: 1.7720 - metric: 0.519 - time: 142.0 (0.9) 
best
Epoch 14/125
train loss: 1.8271 - metric: 0.506 - time: 152.2 (10.2) 
valid loss: 1.7516 - metric: 0.526 - time: 152.2 (0.8) 
best
Epoch 15/125
train loss: 1.8168 - metric: 0.508 - time: 162.5 (10.3) 
valid loss: 1.7657 - metric: 0.519 - time: 162.5 (0.9) 
Epoch 16/125
train loss: 1.8133 - metric: 0.509 - time: 172.6 (10.1) 
valid loss: 1.7392 - metric: 0.524 - time: 172.6 (0.9) 
Epoch 17/125
train loss: 1.8007 - metric: 0.512 - time: 183.5 (10.9) 
valid loss: 1.7375 - metric: 0.529 - time: 183.5 (0.9) 
best
Epoch 18/125
train loss: 1.7958 - metric: 0.513 - time: 193.6 (10.2) 
valid loss: 1.7454 - metric: 0.524 - time: 193.6 (0.9) 
Epoch 19/125
train loss: 1.7911 - metric: 0.514 - time: 203.9 (10.2) 
valid loss: 1.7191 - metric: 0.533 - time: 203.9 (0.9) 
best
Epoch 20/125
train loss: 1.7898 - metric: 0.514 - time: 214.0 (10.2) 
valid loss: 1.7330 - metric: 0.526 - time: 214.0 (0.9) 
Epoch 21/125
train loss: 1.7897 - metric: 0.514 - time: 224.1 (10.1) 
valid loss: 1.7267 - metric: 0.528 - time: 224.1 (0.9) 
Epoch 22/125
train loss: 1.7810 - metric: 0.516 - time: 234.5 (10.3) 
valid loss: 1.7051 - metric: 0.532 - time: 234.5 (0.8) 
Epoch 23/125
train loss: 1.7767 - metric: 0.517 - time: 244.9 (10.5) 
valid loss: 1.7379 - metric: 0.530 - time: 244.9 (0.9) 
Epoch 24/125
train loss: 1.7821 - metric: 0.515 - time: 255.4 (10.4) 
valid loss: 1.7394 - metric: 0.530 - time: 255.4 (0.9) 
Epoch 25/125
train loss: 1.7709 - metric: 0.517 - time: 265.6 (10.3) 
valid loss: 1.7234 - metric: 0.528 - time: 265.6 (0.9) 
Epoch 26/125
train loss: 1.7699 - metric: 0.519 - time: 275.8 (10.2) 
valid loss: 1.6938 - metric: 0.536 - time: 275.8 (0.9) 
best
Epoch 27/125
train loss: 1.7660 - metric: 0.519 - time: 286.0 (10.2) 
valid loss: 1.7045 - metric: 0.532 - time: 286.0 (0.9) 
Epoch 28/125
train loss: 1.7643 - metric: 0.518 - time: 296.3 (10.3) 
valid loss: 1.7079 - metric: 0.534 - time: 296.3 (0.9) 
Epoch 29/125
train loss: 1.7600 - metric: 0.520 - time: 306.5 (10.2) 
valid loss: 1.7105 - metric: 0.534 - time: 306.5 (0.9) 
Epoch 30/125
train loss: 1.7594 - metric: 0.521 - time: 316.8 (10.2) 
valid loss: 1.7192 - metric: 0.534 - time: 316.8 (0.9) 
Epoch 31/125
train loss: 1.7597 - metric: 0.521 - time: 327.0 (10.2) 
valid loss: 1.6934 - metric: 0.533 - time: 327.0 (0.9) 
Epoch 32/125
train loss: 1.7556 - metric: 0.522 - time: 337.1 (10.1) 
valid loss: 1.6937 - metric: 0.539 - time: 337.1 (0.8) 
best
Epoch 33/125
train loss: 1.7557 - metric: 0.522 - time: 347.4 (10.2) 
valid loss: 1.6907 - metric: 0.535 - time: 347.4 (0.9) 
Epoch 34/125
train loss: 1.7559 - metric: 0.521 - time: 358.1 (10.7) 
valid loss: 1.6844 - metric: 0.538 - time: 358.1 (0.9) 
Epoch 35/125
train loss: 1.7570 - metric: 0.521 - time: 368.4 (10.3) 
valid loss: 1.7036 - metric: 0.533 - time: 368.4 (0.9) 
Epoch 36/125
train loss: 1.7471 - metric: 0.523 - time: 378.6 (10.2) 
valid loss: 1.7111 - metric: 0.534 - time: 378.6 (0.9) 
Epoch 37/125
train loss: 1.7513 - metric: 0.522 - time: 388.9 (10.3) 
valid loss: 1.6908 - metric: 0.538 - time: 388.9 (0.9) 
Epoch 38/125
train loss: 1.7442 - metric: 0.524 - time: 399.0 (10.2) 
valid loss: 1.7014 - metric: 0.539 - time: 399.0 (0.9) 
Epoch 39/125
train loss: 1.7431 - metric: 0.524 - time: 409.7 (10.7) 
valid loss: 1.6796 - metric: 0.540 - time: 409.7 (0.9) 
best
Epoch 40/125
train loss: 1.7457 - metric: 0.524 - time: 420.1 (10.3) 
valid loss: 1.6834 - metric: 0.538 - time: 420.1 (0.9) 
Epoch 41/125
"""
