import os

from easydict import EasyDict
from configs.data_cfg import cfg as data_cfg
from configs.model_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()
cfg.seed = 0

cfg.batch_size = 128  # 16
cfg.lr = 3e-3  # 1 3 5
cfg.lr_decay_factor = 1e-3
cfg.lr_decay_by_block = 0.9

cfg.weight_decay = 1e-5
cfg.target = 'num'
cfg.num_masks = 1

cfg.loss = 'L1Loss'  # CrossEntropyLoss SmoothL1Loss KLDivLoss L1Loss
cfg.loss_args = {}  # 'reduction': 'batchmean'}
cfg.decay = 'linear'

cfg.accelerator_args = {'mixed_precision': 'fp16', 'cpu': True}

model_cfg.pred_dim = 1  # cfg.num_embed_features[0]  # 1  #
cfg.model_cfg = model_cfg
cfg.data_cfg = data_cfg

cfg.exp_dir = os.path.join(ROOT_DIR, 'exp_dir')  # , 'train')
cfg.load_pretrained = os.path.join(ROOT_DIR, 'exp_dir', 'MaskedTableModeling.pt')
# 'MaskedTableAutoencoder.pt')  # 'pretrain', "transformer.pt")

cfg.num_epoch = 125
cfg.model = 'PricePrediction'


"""
mae
cpu
Epoch 1/125
train loss: 0.6930 - metric: 0.441 - time: 3.6 (3.6) 
valid loss: 0.3730 - metric: 0.208 - time: 3.6 (0.2) 
best
Epoch 2/125
train loss: 0.3194 - metric: 0.173 - time: 7.1 (3.5) 
valid loss: 0.2412 - metric: 0.120 - time: 7.1 (0.2) 
best
Epoch 3/125
train loss: 0.2358 - metric: 0.128 - time: 11.2 (4.1) 
valid loss: 0.2063 - metric: 0.099 - time: 11.2 (0.2) 
best
Epoch 4/125
train loss: 0.2004 - metric: 0.109 - time: 14.8 (3.6) 
valid loss: 0.1940 - metric: 0.095 - time: 14.8 (0.2) 
best
Epoch 5/125
train loss: 0.1842 - metric: 0.100 - time: 19.0 (4.2) 
valid loss: 0.1717 - metric: 0.085 - time: 19.0 (0.2) 
best
Epoch 6/125
train loss: 0.1762 - metric: 0.096 - time: 22.7 (3.7) 
valid loss: 0.1695 - metric: 0.086 - time: 22.7 (0.2) 
Epoch 7/125
train loss: 0.1676 - metric: 0.091 - time: 26.9 (4.2) 
valid loss: 0.1713 - metric: 0.088 - time: 26.9 (0.2) 
Epoch 8/125
train loss: 0.1576 - metric: 0.086 - time: 30.7 (3.8) 
valid loss: 0.1505 - metric: 0.074 - time: 30.7 (0.2) 
best
Epoch 9/125
train loss: 0.1525 - metric: 0.083 - time: 34.3 (3.7) 
valid loss: 0.1655 - metric: 0.079 - time: 34.3 (0.2) 
Epoch 10/125
train loss: 0.1472 - metric: 0.080 - time: 38.2 (3.9) 
valid loss: 0.1550 - metric: 0.074 - time: 38.2 (0.2) 
best
Epoch 11/125
train loss: 0.1439 - metric: 0.078 - time: 41.7 (3.6) 
valid loss: 0.1427 - metric: 0.071 - time: 41.7 (0.2) 
best
Epoch 12/125
train loss: 0.1408 - metric: 0.077 - time: 45.2 (3.5) 
valid loss: 0.1436 - metric: 0.071 - time: 45.2 (0.2) 
Epoch 13/125
train loss: 0.1397 - metric: 0.076 - time: 48.8 (3.5) 
valid loss: 0.1560 - metric: 0.075 - time: 48.8 (0.2) 
Epoch 14/125
train loss: 0.1377 - metric: 0.075 - time: 52.3 (3.5) 
valid loss: 0.1427 - metric: 0.070 - time: 52.3 (0.2) 
best
Epoch 15/125
train loss: 0.1340 - metric: 0.073 - time: 55.8 (3.5) 
valid loss: 0.1418 - metric: 0.071 - time: 55.8 (0.2) 
Epoch 16/125
train loss: 0.1313 - metric: 0.072 - time: 59.3 (3.5) 
valid loss: 0.1358 - metric: 0.067 - time: 59.3 (0.2) 
best
Epoch 17/125
train loss: 0.1299 - metric: 0.071 - time: 62.9 (3.5) 
valid loss: 0.1452 - metric: 0.069 - time: 62.9 (0.2) 
Epoch 18/125
train loss: 0.1273 - metric: 0.069 - time: 66.4 (3.5) 
valid loss: 0.1340 - metric: 0.065 - time: 66.4 (0.2) 
best
Epoch 19/125
train loss: 0.1260 - metric: 0.069 - time: 69.9 (3.5) 
valid loss: 0.1396 - metric: 0.067 - time: 69.9 (0.2) 
Epoch 20/125
train loss: 0.1239 - metric: 0.067 - time: 73.4 (3.5) 
valid loss: 0.1314 - metric: 0.064 - time: 73.4 (0.2) 
best
Epoch 21/125
train loss: 0.1234 - metric: 0.067 - time: 77.0 (3.5) 
valid loss: 0.1418 - metric: 0.068 - time: 77.0 (0.2) 
Epoch 22/125
train loss: 0.1215 - metric: 0.066 - time: 80.5 (3.5) 
valid loss: 0.1370 - metric: 0.065 - time: 80.5 (0.2) 
Epoch 23/125
train loss: 0.1209 - metric: 0.066 - time: 84.0 (3.5) 
valid loss: 0.1308 - metric: 0.063 - time: 84.0 (0.2) 
best
Epoch 24/125
train loss: 0.1193 - metric: 0.065 - time: 87.6 (3.5) 
valid loss: 0.1437 - metric: 0.069 - time: 87.6 (0.2) 
Epoch 25/125
train loss: 0.1194 - metric: 0.065 - time: 91.1 (3.5) 
valid loss: 0.1405 - metric: 0.068 - time: 91.1 (0.2) 
Epoch 26/125
train loss: 0.1174 - metric: 0.064 - time: 94.7 (3.6) 
valid loss: 0.1376 - metric: 0.067 - time: 94.7 (0.2) 
Epoch 27/125
train loss: 0.1163 - metric: 0.063 - time: 98.3 (3.6) 
valid loss: 0.1315 - metric: 0.063 - time: 98.3 (0.2) 
Epoch 28/125
train loss: 0.1149 - metric: 0.063 - time: 101.9 (3.6) 
valid loss: 0.1406 - metric: 0.065 - time: 101.9 (0.2) 
Epoch 29/125
train loss: 0.1146 - metric: 0.062 - time: 105.5 (3.6) 
valid loss: 0.1261 - metric: 0.061 - time: 105.5 (0.2) 
best
Epoch 30/125
train loss: 0.1138 - metric: 0.062 - time: 109.1 (3.6) 
valid loss: 0.1313 - metric: 0.063 - time: 109.1 (0.2) 
Epoch 31/125


mtm
cpu
Epoch 1/125
train loss: 0.7069 - metric: 0.450 - time: 4.0 (4.0) 
valid loss: 0.3758 - metric: 0.207 - time: 4.0 (0.2) 
best
Epoch 2/125
train loss: 0.3278 - metric: 0.177 - time: 7.6 (3.6) 
valid loss: 0.2450 - metric: 0.117 - time: 7.6 (0.2) 
best
Epoch 3/125
train loss: 0.2435 - metric: 0.131 - time: 11.2 (3.6) 
valid loss: 0.2063 - metric: 0.109 - time: 11.2 (0.2) 
best
Epoch 4/125
train loss: 0.2115 - metric: 0.115 - time: 14.8 (3.6) 
valid loss: 0.1992 - metric: 0.096 - time: 14.8 (0.2) 
best
Epoch 5/125
train loss: 0.2030 - metric: 0.110 - time: 18.6 (3.8) 
valid loss: 0.2163 - metric: 0.101 - time: 18.6 (0.2) 
Epoch 6/125
train loss: 0.1850 - metric: 0.100 - time: 22.3 (3.8) 
valid loss: 0.1793 - metric: 0.089 - time: 22.3 (0.2) 
best
Epoch 7/125
train loss: 0.1748 - metric: 0.095 - time: 26.1 (3.8) 
valid loss: 0.1733 - metric: 0.085 - time: 26.1 (0.2) 
best
Epoch 8/125
train loss: 0.1657 - metric: 0.090 - time: 29.8 (3.7) 
valid loss: 0.1566 - metric: 0.075 - time: 29.8 (0.2) 
best
Epoch 9/125
train loss: 0.1580 - metric: 0.086 - time: 33.6 (3.8) 
valid loss: 0.1482 - metric: 0.072 - time: 33.6 (0.2) 
best
Epoch 10/125
train loss: 0.1535 - metric: 0.083 - time: 37.3 (3.7) 
valid loss: 0.1544 - metric: 0.074 - time: 37.3 (0.2) 
Epoch 11/125
train loss: 0.1502 - metric: 0.082 - time: 41.0 (3.7) 
valid loss: 0.1534 - metric: 0.072 - time: 41.0 (0.2) 
Epoch 12/125
train loss: 0.1464 - metric: 0.080 - time: 44.6 (3.6) 
valid loss: 0.1560 - metric: 0.074 - time: 44.6 (0.2) 
Epoch 13/125
train loss: 0.1439 - metric: 0.079 - time: 48.2 (3.7) 
valid loss: 0.1448 - metric: 0.069 - time: 48.2 (0.2) 
best
Epoch 14/125
train loss: 0.1415 - metric: 0.077 - time: 51.9 (3.6) 
valid loss: 0.1506 - metric: 0.071 - time: 51.9 (0.2) 
Epoch 15/125
train loss: 0.1374 - metric: 0.075 - time: 55.5 (3.7) 
valid loss: 0.1663 - metric: 0.079 - time: 55.5 (0.2) 
Epoch 16/125
train loss: 0.1358 - metric: 0.074 - time: 59.3 (3.7) 
valid loss: 0.1448 - metric: 0.069 - time: 59.3 (0.2) 
best
Epoch 17/125
train loss: 0.1334 - metric: 0.073 - time: 62.9 (3.7) 
valid loss: 0.1477 - metric: 0.070 - time: 62.9 (0.2) 
Epoch 18/125
train loss: 0.1313 - metric: 0.072 - time: 66.6 (3.7) 
valid loss: 0.1412 - metric: 0.069 - time: 66.6 (0.2) 
best
Epoch 19/125
train loss: 0.1290 - metric: 0.070 - time: 70.2 (3.6) 
valid loss: 0.1366 - metric: 0.065 - time: 70.2 (0.2) 
best
Epoch 20/125
train loss: 0.1291 - metric: 0.071 - time: 73.8 (3.6) 
valid loss: 0.1354 - metric: 0.068 - time: 73.8 (0.2) 
Epoch 21/125
train loss: 0.1265 - metric: 0.069 - time: 77.3 (3.5) 
valid loss: 0.1344 - metric: 0.064 - time: 77.3 (0.2) 
best
Epoch 22/125
train loss: 0.1257 - metric: 0.068 - time: 80.9 (3.5) 
valid loss: 0.1521 - metric: 0.073 - time: 80.9 (0.2) 
Epoch 23/125
train loss: 0.1239 - metric: 0.068 - time: 84.5 (3.6) 
valid loss: 0.1424 - metric: 0.066 - time: 84.5 (0.2) 
Epoch 24/125
train loss: 0.1225 - metric: 0.067 - time: 88.0 (3.6) 
valid loss: 0.1303 - metric: 0.061 - time: 88.0 (0.2) 
best
Epoch 25/125
train loss: 0.1224 - metric: 0.067 - time: 91.6 (3.6) 
valid loss: 0.1356 - metric: 0.064 - time: 91.6 (0.2) 
Epoch 26/125
train loss: 0.1196 - metric: 0.065 - time: 95.2 (3.6) 
valid loss: 0.1278 - metric: 0.061 - time: 95.2 (0.2) 
best
Epoch 27/125
train loss: 0.1187 - metric: 0.065 - time: 98.7 (3.5) 
valid loss: 0.1289 - metric: 0.061 - time: 98.7 (0.2) 
Epoch 28/125
train loss: 0.1181 - metric: 0.064 - time: 102.3 (3.6) 
valid loss: 0.1336 - metric: 0.061 - time: 102.3 (0.2) 
Epoch 29/125
train loss: 0.1170 - metric: 0.064 - time: 106.2 (3.9) 
valid loss: 0.1411 - metric: 0.068 - time: 106.2 (0.2) 
Epoch 30/125
train loss: 0.1153 - metric: 0.063 - time: 109.7 (3.6) 
valid loss: 0.1283 - metric: 0.060 - time: 109.7 (0.2) 
best
Epoch 31/125
"""