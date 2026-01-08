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
cfg.num_masks = int(0.4 * len(data_cfg.features))


"""
mae
cpu
Epoch 1/125
train loss: 6.5060 - metric: 0.108 - time: 9.8 (9.8) 
valid loss: 5.3773 - metric: 0.374 - time: 9.8 (0.9) 
best
Epoch 2/125
train loss: 4.0992 - metric: 0.383 - time: 19.4 (9.6) 
valid loss: 2.8704 - metric: 0.401 - time: 19.4 (0.9) 
best
Epoch 3/125
train loss: 2.5460 - metric: 0.408 - time: 29.3 (9.9) 
valid loss: 2.3336 - metric: 0.419 - time: 29.3 (1.0) 
best
Epoch 4/125
train loss: 2.2718 - metric: 0.427 - time: 39.0 (9.7) 
valid loss: 2.1863 - metric: 0.437 - time: 39.0 (0.9) 
best
Epoch 5/125
train loss: 2.1570 - metric: 0.445 - time: 49.6 (10.6) 
valid loss: 2.0752 - metric: 0.457 - time: 49.6 (1.1) 
best
Epoch 6/125
train loss: 2.0752 - metric: 0.457 - time: 60.3 (10.6) 
valid loss: 2.0146 - metric: 0.468 - time: 60.3 (0.9) 
best
Epoch 7/125
train loss: 2.0251 - metric: 0.466 - time: 70.1 (9.8) 
valid loss: 1.9652 - metric: 0.480 - time: 70.1 (1.0) 
best
Epoch 8/125
train loss: 1.9893 - metric: 0.473 - time: 79.8 (9.8) 
valid loss: 1.9337 - metric: 0.485 - time: 79.8 (1.0) 
best
Epoch 9/125
train loss: 1.9571 - metric: 0.479 - time: 90.7 (10.9) 
valid loss: 1.9198 - metric: 0.488 - time: 90.7 (1.0) 
best
Epoch 10/125
train loss: 1.9364 - metric: 0.484 - time: 101.1 (10.4) 
valid loss: 1.8777 - metric: 0.496 - time: 101.1 (0.9) 
best
Epoch 11/125
train loss: 1.9217 - metric: 0.487 - time: 111.9 (10.8) 
valid loss: 1.8671 - metric: 0.497 - time: 111.9 (1.0) 
best
Epoch 12/125
train loss: 1.9018 - metric: 0.492 - time: 122.1 (10.2) 
valid loss: 1.8512 - metric: 0.500 - time: 122.1 (1.0) 
best
Epoch 13/125
train loss: 1.8880 - metric: 0.494 - time: 132.3 (10.3) 
valid loss: 1.8400 - metric: 0.502 - time: 132.3 (1.0) 
best
Epoch 14/125
train loss: 1.8797 - metric: 0.496 - time: 142.8 (10.5) 
valid loss: 1.8138 - metric: 0.507 - time: 142.8 (1.0) 
best
Epoch 15/125
train loss: 1.8683 - metric: 0.498 - time: 153.0 (10.2) 
valid loss: 1.8124 - metric: 0.510 - time: 153.0 (1.0) 
best
Epoch 16/125
train loss: 1.8664 - metric: 0.499 - time: 163.4 (10.4) 
valid loss: 1.8274 - metric: 0.509 - time: 163.4 (0.9) 
Epoch 17/125
train loss: 1.8567 - metric: 0.500 - time: 173.7 (10.3) 
valid loss: 1.8161 - metric: 0.508 - time: 173.7 (1.0) 
Epoch 18/125
train loss: 1.8542 - metric: 0.500 - time: 184.0 (10.3) 
valid loss: 1.8040 - metric: 0.511 - time: 184.0 (0.9) 
best
Epoch 19/125
train loss: 1.8466 - metric: 0.502 - time: 195.6 (11.6) 
valid loss: 1.8194 - metric: 0.515 - time: 195.6 (2.2) 
best
Epoch 20/125
train loss: 1.8383 - metric: 0.504 - time: 207.8 (12.2) 
valid loss: 1.7876 - metric: 0.513 - time: 207.8 (1.0) 
Epoch 21/125
train loss: 1.8360 - metric: 0.504 - time: 217.9 (10.1) 
valid loss: 1.7728 - metric: 0.521 - time: 217.9 (0.9) 
best
Epoch 22/125
train loss: 1.8279 - metric: 0.506 - time: 227.9 (10.0) 
valid loss: 1.7793 - metric: 0.517 - time: 227.9 (0.9) 
Epoch 23/125
train loss: 1.8285 - metric: 0.507 - time: 238.0 (10.1) 
valid loss: 1.7704 - metric: 0.520 - time: 238.0 (0.9) 
Epoch 24/125
train loss: 1.8145 - metric: 0.510 - time: 248.1 (10.1) 
valid loss: 1.7809 - metric: 0.518 - time: 248.1 (1.0) 
Epoch 25/125
train loss: 1.8205 - metric: 0.507 - time: 258.1 (10.0) 
valid loss: 1.7703 - metric: 0.520 - time: 258.1 (0.9) 
Epoch 26/125
train loss: 1.8164 - metric: 0.508 - time: 268.0 (10.0) 
valid loss: 1.7709 - metric: 0.524 - time: 268.0 (0.9) 
best
Epoch 27/125
train loss: 1.8104 - metric: 0.511 - time: 278.1 (10.1) 
valid loss: 1.7607 - metric: 0.524 - time: 278.1 (1.0) 
Epoch 28/125
train loss: 1.8158 - metric: 0.509 - time: 290.9 (12.8) 
valid loss: 1.7515 - metric: 0.522 - time: 290.9 (1.0) 
Epoch 29/125
train loss: 1.8131 - metric: 0.509 - time: 301.3 (10.3) 
valid loss: 1.7662 - metric: 0.521 - time: 301.3 (1.0) 
Epoch 30/125
train loss: 1.8033 - metric: 0.511 - time: 311.3 (10.0) 
valid loss: 1.7732 - metric: 0.524 - time: 311.3 (0.9) 
Epoch 31/125
train loss: 1.8058 - metric: 0.511 - time: 321.1 (9.8) 
valid loss: 1.7713 - metric: 0.521 - time: 321.1 (0.9) 
Epoch 32/125
train loss: 1.7998 - metric: 0.513 - time: 331.1 (10.0) 
valid loss: 1.7552 - metric: 0.529 - time: 331.1 (0.9) 
best
Epoch 33/125
train loss: 1.7966 - metric: 0.513 - time: 341.0 (10.0) 
valid loss: 1.7394 - metric: 0.532 - time: 341.0 (1.0) 
best
Epoch 34/125
train loss: 1.7990 - metric: 0.513 - time: 351.1 (10.1) 
valid loss: 1.7666 - metric: 0.522 - time: 351.1 (0.9) 
Epoch 35/125
train loss: 1.7958 - metric: 0.514 - time: 361.3 (10.2) 
valid loss: 1.7799 - metric: 0.520 - time: 361.3 (1.0) 
Epoch 36/125
train loss: 1.7952 - metric: 0.514 - time: 371.3 (10.0) 
valid loss: 1.7559 - metric: 0.526 - time: 371.3 (1.0) 
Epoch 37/125
train loss: 1.7943 - metric: 0.514 - time: 381.4 (10.0) 
valid loss: 1.7423 - metric: 0.525 - time: 381.4 (1.0) 
Epoch 38/125
train loss: 1.7916 - metric: 0.513 - time: 391.4 (10.0) 
valid loss: 1.7523 - metric: 0.525 - time: 391.4 (0.9) 
Epoch 39/125
train loss: 1.7874 - metric: 0.515 - time: 401.5 (10.1) 
valid loss: 1.7496 - metric: 0.526 - time: 401.5 (1.1) 
Epoch 40/125
train loss: 1.7917 - metric: 0.515 - time: 411.7 (10.1) 
valid loss: 1.7572 - metric: 0.524 - time: 411.7 (1.0) 
Epoch 41/125


mtm
cpu
Epoch 1/125
train loss: 6.1983 - metric: 0.137 - time: 9.8 (9.8) 
valid loss: 4.3920 - metric: 0.388 - time: 9.8 (0.7) 
best
Epoch 2/125
train loss: 3.1659 - metric: 0.402 - time: 18.0 (8.2) 
valid loss: 2.4399 - metric: 0.410 - time: 18.0 (0.7) 
best
Epoch 3/125
train loss: 2.2917 - metric: 0.441 - time: 26.1 (8.1) 
valid loss: 2.0849 - metric: 0.485 - time: 26.1 (0.7) 
best
Epoch 4/125
train loss: 2.0130 - metric: 0.501 - time: 34.2 (8.1) 
valid loss: 1.8067 - metric: 0.539 - time: 34.2 (0.7) 
best
Epoch 5/125
train loss: 1.7935 - metric: 0.540 - time: 42.3 (8.2) 
valid loss: 1.6111 - metric: 0.571 - time: 42.3 (0.7) 
best
Epoch 6/125
train loss: 1.6692 - metric: 0.563 - time: 50.4 (8.0) 
valid loss: 1.5345 - metric: 0.590 - time: 50.4 (0.7) 
best
Epoch 7/125
train loss: 1.5970 - metric: 0.577 - time: 58.5 (8.1) 
valid loss: 1.4839 - metric: 0.602 - time: 58.5 (0.7) 
best
Epoch 8/125
train loss: 1.5504 - metric: 0.586 - time: 67.0 (8.5) 
valid loss: 1.4377 - metric: 0.618 - time: 67.0 (0.7) 
best
Epoch 9/125
train loss: 1.5122 - metric: 0.593 - time: 75.4 (8.4) 
valid loss: 1.4206 - metric: 0.623 - time: 75.4 (0.7) 
best
Epoch 10/125
train loss: 1.4901 - metric: 0.600 - time: 84.0 (8.6) 
valid loss: 1.3701 - metric: 0.624 - time: 84.0 (0.7) 
best
Epoch 11/125
train loss: 1.4692 - metric: 0.603 - time: 92.3 (8.3) 
valid loss: 1.3647 - metric: 0.631 - time: 92.3 (0.8) 
best
Epoch 12/125
train loss: 1.4549 - metric: 0.608 - time: 101.2 (8.9) 
valid loss: 1.3519 - metric: 0.631 - time: 101.2 (0.8) 
best
Epoch 13/125
train loss: 1.4405 - metric: 0.611 - time: 110.2 (8.9) 
valid loss: 1.3517 - metric: 0.637 - time: 110.2 (0.8) 
best
Epoch 14/125
train loss: 1.4315 - metric: 0.613 - time: 119.0 (8.8) 
valid loss: 1.3369 - metric: 0.636 - time: 119.0 (0.8) 
Epoch 15/125
train loss: 1.4194 - metric: 0.615 - time: 127.8 (8.8) 
valid loss: 1.3246 - metric: 0.638 - time: 127.8 (0.8) 
best
Epoch 16/125
train loss: 1.4046 - metric: 0.619 - time: 137.0 (9.2) 
valid loss: 1.3051 - metric: 0.642 - time: 137.0 (0.8) 
best
Epoch 17/125
train loss: 1.4074 - metric: 0.618 - time: 145.9 (8.8) 
valid loss: 1.3091 - metric: 0.644 - time: 145.9 (0.8) 
best
Epoch 18/125
train loss: 1.4001 - metric: 0.620 - time: 155.2 (9.3) 
valid loss: 1.3242 - metric: 0.643 - time: 155.2 (0.8) 
Epoch 19/125
train loss: 1.3885 - metric: 0.622 - time: 164.2 (9.0) 
valid loss: 1.2968 - metric: 0.644 - time: 164.2 (0.8) 
Epoch 20/125
train loss: 1.3861 - metric: 0.624 - time: 173.1 (8.9) 
valid loss: 1.2960 - metric: 0.650 - time: 173.1 (0.8) 
best
Epoch 21/125
train loss: 1.3881 - metric: 0.623 - time: 182.0 (8.9) 
valid loss: 1.3114 - metric: 0.648 - time: 182.0 (0.8) 
Epoch 22/125
train loss: 1.3849 - metric: 0.623 - time: 191.1 (9.1) 
valid loss: 1.2740 - metric: 0.650 - time: 191.1 (0.8) 
best
Epoch 23/125
train loss: 1.3729 - metric: 0.626 - time: 200.5 (9.4) 
valid loss: 1.3059 - metric: 0.653 - time: 200.5 (0.8) 
best
Epoch 24/125
train loss: 1.3688 - metric: 0.628 - time: 209.4 (8.9) 
valid loss: 1.2839 - metric: 0.649 - time: 209.4 (0.8) 
Epoch 25/125
train loss: 1.3721 - metric: 0.626 - time: 218.2 (8.8) 
valid loss: 1.2797 - metric: 0.648 - time: 218.2 (0.8) 
Epoch 26/125
train loss: 1.3599 - metric: 0.630 - time: 227.1 (8.8) 
valid loss: 1.2973 - metric: 0.653 - time: 227.1 (0.8) 
best
Epoch 27/125
train loss: 1.3670 - metric: 0.627 - time: 235.9 (8.9) 
valid loss: 1.2855 - metric: 0.652 - time: 235.9 (0.8) 
Epoch 28/125
train loss: 1.3590 - metric: 0.630 - time: 245.2 (9.2) 
valid loss: 1.2741 - metric: 0.655 - time: 245.2 (0.9) 
best
Epoch 29/125
train loss: 1.3571 - metric: 0.629 - time: 254.1 (8.9) 
valid loss: 1.2913 - metric: 0.650 - time: 254.1 (0.8) 
Epoch 30/125
train loss: 1.3530 - metric: 0.631 - time: 262.9 (8.8) 
valid loss: 1.2590 - metric: 0.655 - time: 262.9 (0.8) 
Epoch 31/125
train loss: 1.3517 - metric: 0.632 - time: 271.8 (8.9) 
valid loss: 1.2792 - metric: 0.656 - time: 271.8 (0.8) 
best
Epoch 32/125
train loss: 1.3477 - metric: 0.631 - time: 280.7 (8.9) 
valid loss: 1.2703 - metric: 0.654 - time: 280.7 (0.8) 
Epoch 33/125
train loss: 1.3424 - metric: 0.633 - time: 289.6 (8.8) 
valid loss: 1.2551 - metric: 0.661 - time: 289.6 (0.8) 
best
Epoch 34/125
train loss: 1.3462 - metric: 0.632 - time: 299.2 (9.6) 
valid loss: 1.2567 - metric: 0.656 - time: 299.2 (0.9) 
Epoch 35/125
train loss: 1.3387 - metric: 0.634 - time: 308.3 (9.1) 
valid loss: 1.2556 - metric: 0.659 - time: 308.3 (0.8) 
Epoch 36/125
train loss: 1.3407 - metric: 0.633 - time: 317.1 (8.9) 
valid loss: 1.2562 - metric: 0.655 - time: 317.1 (0.8) 
Epoch 37/125
train loss: 1.3390 - metric: 0.634 - time: 325.9 (8.8) 
valid loss: 1.2763 - metric: 0.656 - time: 325.9 (0.8) 
Epoch 38/125
train loss: 1.3400 - metric: 0.634 - time: 334.8 (8.9) 
valid loss: 1.2874 - metric: 0.657 - time: 334.8 (0.8) 
Epoch 39/125
train loss: 1.3373 - metric: 0.634 - time: 343.6 (8.8) 
valid loss: 1.2544 - metric: 0.658 - time: 343.6 (0.8) 
Epoch 40/125
train loss: 1.3308 - metric: 0.635 - time: 352.7 (9.1) 
valid loss: 1.2620 - metric: 0.657 - time: 352.7 (0.8) 
Epoch 41/125
"""
