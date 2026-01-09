import os

from easydict import EasyDict
from configs.data_cfg import cfg as data_cfg
from configs.model_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()
cfg.seed = 0

cfg.batch_size = 1024  # 16
cfg.lr = 3e-3  # 1 3 5
cfg.lr_decay_factor = 1e-3
# cfg.lr_decay_by_block = 0.5

cfg.weight_decay = 1e-1
cfg.target = 'num'  # cat
cfg.num_masks = 1

cfg.loss = 'MSELoss'  # CrossEntropyLoss SmoothL1Loss KLDivLoss L1Loss
cfg.loss_args = {}  # 'reduction': 'batchmean'}
cfg.decay = 'cosine'

cfg.accelerator_args = {'mixed_precision': 'fp16', 'cpu': True}

model_cfg.pred_dim = 1  # model_cfg.num_embed_features[0]  # 1  #
cfg.model_cfg = model_cfg
cfg.data_cfg = data_cfg

cfg.exp_dir = os.path.join(ROOT_DIR, 'exp_dir')  # , 'train')
# cfg.load_pretrained = os.path.join(ROOT_DIR, 'exp_dir', 'MaskedTableModeling.pt')
# 'MaskedTableAutoencoder.pt')  # 'pretrain', "transformer.pt") MaskedTableModeling

cfg.num_epoch = 125
cfg.model = 'PricePrediction'


"""
cpu
Epoch 1/125
train loss: 0.3860 - metric: 0.462 - time: 5.2 (5.2) 
valid loss: 0.2160 - metric: 0.345 - time: 5.2 (0.3) 
best
Epoch 2/125
train loss: 0.1781 - metric: 0.258 - time: 10.5 (5.4) 
valid loss: 0.1092 - metric: 0.168 - time: 10.5 (0.2) 
best
Epoch 3/125
train loss: 0.0769 - metric: 0.160 - time: 15.6 (5.1) 
valid loss: 0.1109 - metric: 0.124 - time: 15.6 (0.3) 
best
Epoch 4/125
train loss: 0.0492 - metric: 0.126 - time: 20.7 (5.1) 
valid loss: 0.1120 - metric: 0.098 - time: 20.7 (0.3) 
best
Epoch 5/125
train loss: 0.0387 - metric: 0.113 - time: 25.7 (5.0) 
valid loss: 0.1053 - metric: 0.094 - time: 25.7 (0.3) 
best
Epoch 6/125
train loss: 0.0321 - metric: 0.102 - time: 30.9 (5.1) 
valid loss: 0.0942 - metric: 0.084 - time: 30.9 (0.3) 
best
Epoch 7/125
train loss: 0.0279 - metric: 0.095 - time: 36.0 (5.1) 
valid loss: 0.0884 - metric: 0.080 - time: 36.0 (0.3) 
best
Epoch 8/125
train loss: 0.0249 - metric: 0.090 - time: 41.1 (5.1) 
valid loss: 0.0848 - metric: 0.080 - time: 41.1 (0.3) 
best
Epoch 9/125
train loss: 0.0226 - metric: 0.086 - time: 46.1 (5.0) 
valid loss: 0.1055 - metric: 0.079 - time: 46.1 (0.3) 
best
Epoch 10/125
train loss: 0.0207 - metric: 0.081 - time: 51.2 (5.0) 
valid loss: 0.1041 - metric: 0.078 - time: 51.2 (0.3) 
best
Epoch 11/125
train loss: 0.0194 - metric: 0.079 - time: 56.2 (5.1) 
valid loss: 0.0933 - metric: 0.074 - time: 56.2 (0.3) 
best
Epoch 12/125
train loss: 0.0185 - metric: 0.077 - time: 61.3 (5.0) 
valid loss: 0.0880 - metric: 0.069 - time: 61.3 (0.3) 
best
Epoch 13/125
train loss: 0.0178 - metric: 0.076 - time: 66.3 (5.1) 
valid loss: 0.0908 - metric: 0.066 - time: 66.3 (0.3) 
best
Epoch 14/125
train loss: 0.0174 - metric: 0.075 - time: 71.5 (5.1) 
valid loss: 0.0738 - metric: 0.067 - time: 71.5 (0.3) 
Epoch 15/125
train loss: 0.0164 - metric: 0.073 - time: 76.5 (5.1) 
valid loss: 0.0777 - metric: 0.065 - time: 76.5 (0.3) 
best
Epoch 16/125
train loss: 0.0157 - metric: 0.071 - time: 81.6 (5.1) 
valid loss: 0.0724 - metric: 0.062 - time: 81.6 (0.3) 
best
Epoch 17/125
train loss: 0.0148 - metric: 0.069 - time: 86.8 (5.2) 
valid loss: 0.0782 - metric: 0.064 - time: 86.8 (0.3) 
Epoch 18/125
train loss: 0.0146 - metric: 0.068 - time: 91.8 (5.1) 
valid loss: 0.0865 - metric: 0.066 - time: 91.8 (0.3) 
Epoch 19/125
train loss: 0.0141 - metric: 0.067 - time: 98.3 (6.5) 
valid loss: 0.0875 - metric: 0.066 - time: 98.3 (0.3) 
Epoch 20/125
train loss: 0.0138 - metric: 0.066 - time: 104.7 (6.4) 
valid loss: 0.0896 - metric: 0.069 - time: 104.7 (0.3) 
Epoch 21/125
train loss: 0.0132 - metric: 0.066 - time: 111.5 (6.8) 
valid loss: 0.0880 - metric: 0.066 - time: 111.5 (0.5) 
Epoch 22/125
train loss: 0.0133 - metric: 0.065 - time: 117.0 (5.5) 
valid loss: 0.0831 - metric: 0.060 - time: 117.0 (0.3) 
best
Epoch 23/125
train loss: 0.0128 - metric: 0.064 - time: 124.1 (7.1) 
valid loss: 0.0893 - metric: 0.063 - time: 124.1 (0.4) 
Epoch 24/125
train loss: 0.0123 - metric: 0.063 - time: 129.4 (5.3) 
valid loss: 0.0754 - metric: 0.063 - time: 129.4 (0.3) 
Epoch 25/125
train loss: 0.0119 - metric: 0.062 - time: 134.5 (5.2) 
valid loss: 0.0818 - metric: 0.063 - time: 134.5 (0.3) 
Epoch 26/125
train loss: 0.0113 - metric: 0.061 - time: 139.8 (5.3) 
valid loss: 0.0845 - metric: 0.060 - time: 139.8 (0.3) 
best
Epoch 27/125
train loss: 0.0114 - metric: 0.061 - time: 144.9 (5.1) 
valid loss: 0.0750 - metric: 0.062 - time: 144.9 (0.3) 
Epoch 28/125
train loss: 0.0110 - metric: 0.059 - time: 150.0 (5.1) 
valid loss: 0.0806 - metric: 0.060 - time: 150.0 (0.3) 
best
Epoch 29/125
train loss: 0.0111 - metric: 0.060 - time: 156.4 (6.4) 
valid loss: 0.0778 - metric: 0.060 - time: 156.4 (0.2) 
Epoch 30/125
train loss: 0.0107 - metric: 0.059 - time: 161.4 (5.0) 
valid loss: 0.0752 - metric: 0.060 - time: 161.4 (0.3) 
Epoch 31/125
train loss: 0.0104 - metric: 0.058 - time: 166.5 (5.1) 
valid loss: 0.0790 - metric: 0.059 - time: 166.5 (0.4) 
best
Epoch 32/125
train loss: 0.0100 - metric: 0.057 - time: 172.0 (5.6) 
valid loss: 0.0710 - metric: 0.059 - time: 172.0 (0.4) 
Epoch 33/125
train loss: 0.0098 - metric: 0.057 - time: 178.4 (6.4) 
valid loss: 0.0862 - metric: 0.060 - time: 178.4 (0.3) 
Epoch 34/125
train loss: 0.0096 - metric: 0.056 - time: 183.9 (5.5) 
valid loss: 0.0820 - metric: 0.058 - time: 183.9 (0.3) 
best
Epoch 35/125
train loss: 0.0094 - metric: 0.056 - time: 189.1 (5.2) 
valid loss: 0.0593 - metric: 0.058 - time: 189.1 (0.3) 
Epoch 36/125
train loss: 0.0095 - metric: 0.056 - time: 194.3 (5.2) 
valid loss: 0.0666 - metric: 0.057 - time: 194.3 (0.3) 
best
Epoch 37/125
train loss: 0.0089 - metric: 0.054 - time: 199.4 (5.2) 
valid loss: 0.0750 - metric: 0.060 - time: 199.4 (0.3) 
Epoch 38/125
train loss: 0.0091 - metric: 0.055 - time: 204.7 (5.3) 
valid loss: 0.0626 - metric: 0.057 - time: 204.7 (0.3) 
best
"""