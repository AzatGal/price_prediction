import os

from easydict import EasyDict
from configs.data_cfg import cfg as data_cfg
from configs.model_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()
cfg.seed = 0
cfg.exp_dir = os.path.join(ROOT_DIR, 'exp_dir')

cfg.batch_size = 4*1024  # 16
cfg.num_epoch = 300

cfg.lr = 6e-4  # 1 3 5
cfg.lr_decay_factor = 1e-3

cfg.weight_decay = 0.001
cfg.lr_decay = 'cosine'  # cosine

cfg.loss = 'MSELoss'  # CrossEntropyLoss SmoothL1Loss KLDivLoss L1Loss MSELoss
cfg.loss_args = {}  # 'reduction': 'batchmean'}

cfg.accelerator_args = {'mixed_precision': 'fp16', 'cpu': True}

cfg.target_type = 'num'  # cat num

# cfg.lr_decay_by_block = 0.95
# cfg.load_pretrained = os.path.join(ROOT_DIR, 'exp_dir', 'MaskedTableAutoencoder.pt')
# MaskedTableAutoencoder MaskedTableModeling

cfg.model = 'PricePrediction'  # PricePrediction TabM PricePredEnsemble

model_cfg.pred_dim = 1  # model_cfg.num_embed_features[0]  # 1  #

data_cfg.target_type = cfg.target_type
data_cfg.mask_first_token = len(data_cfg.features) == len(model_cfg.num_embed_features)

cfg.model_cfg = model_cfg
cfg.data_cfg = data_cfg



"""
cpu
Epoch 1/256
train loss: 1.2182 - metric: 0.436 - time: 1.6 (1.6) 
valid loss: 1.1300 - metric: 0.425 - time: 1.6 (0.1) 
best
Epoch 2/256
train loss: 1.0878 - metric: 0.441 - time: 3.0 (1.5) 
valid loss: 0.9559 - metric: 0.447 - time: 3.0 (0.1) 
Epoch 3/256
train loss: 0.9617 - metric: 0.478 - time: 4.5 (1.5) 
valid loss: 0.8784 - metric: 0.481 - time: 4.5 (0.1) 
Epoch 4/256
train loss: 0.8578 - metric: 0.461 - time: 6.0 (1.5) 
valid loss: 0.7305 - metric: 0.392 - time: 6.0 (0.1) 
best
Epoch 5/256
train loss: 0.7223 - metric: 0.371 - time: 7.6 (1.7) 
valid loss: 0.5819 - metric: 0.327 - time: 7.6 (0.1) 
best
Epoch 6/256
train loss: 0.5667 - metric: 0.334 - time: 9.2 (1.6) 
valid loss: 0.4237 - metric: 0.282 - time: 9.2 (0.1) 
best
Epoch 7/256
train loss: 0.4248 - metric: 0.272 - time: 10.7 (1.5) 
valid loss: 0.3093 - metric: 0.241 - time: 10.7 (0.1) 
best
Epoch 8/256
train loss: 0.3244 - metric: 0.237 - time: 12.2 (1.5) 
valid loss: 0.2341 - metric: 0.207 - time: 12.2 (0.1) 
best
Epoch 9/256
train loss: 0.2477 - metric: 0.206 - time: 13.7 (1.5) 
valid loss: 0.1739 - metric: 0.173 - time: 13.7 (0.1) 
best
Epoch 10/256
train loss: 0.1952 - metric: 0.180 - time: 15.2 (1.5) 
valid loss: 0.1358 - metric: 0.148 - time: 15.2 (0.1) 
best
Epoch 11/256
train loss: 0.1601 - metric: 0.161 - time: 16.7 (1.5) 
valid loss: 0.1158 - metric: 0.130 - time: 16.7 (0.1) 
best
Epoch 12/256
train loss: 0.1395 - metric: 0.147 - time: 18.2 (1.5) 
valid loss: 0.1014 - metric: 0.127 - time: 18.2 (0.1) 
best
Epoch 13/256
train loss: 0.1222 - metric: 0.139 - time: 19.7 (1.5) 
valid loss: 0.0892 - metric: 0.116 - time: 19.7 (0.1) 
best
Epoch 14/256
train loss: 0.1087 - metric: 0.130 - time: 21.1 (1.5) 
valid loss: 0.0804 - metric: 0.113 - time: 21.1 (0.1) 
best
Epoch 15/256
train loss: 0.0975 - metric: 0.123 - time: 22.6 (1.5) 
valid loss: 0.0725 - metric: 0.105 - time: 22.6 (0.1) 
best
Epoch 16/256
train loss: 0.0895 - metric: 0.119 - time: 24.1 (1.5) 
valid loss: 0.0664 - metric: 0.102 - time: 24.1 (0.1) 
best
Epoch 17/256
train loss: 0.0832 - metric: 0.113 - time: 25.6 (1.5) 
valid loss: 0.0625 - metric: 0.095 - time: 25.6 (0.1) 
best
Epoch 18/256
train loss: 0.0775 - metric: 0.110 - time: 27.0 (1.5) 
valid loss: 0.0594 - metric: 0.097 - time: 27.0 (0.1) 
Epoch 19/256
train loss: 0.0731 - metric: 0.106 - time: 28.5 (1.5) 
valid loss: 0.0561 - metric: 0.091 - time: 28.5 (0.1) 
best
Epoch 20/256
train loss: 0.0699 - metric: 0.104 - time: 30.0 (1.5) 
valid loss: 0.0537 - metric: 0.091 - time: 30.0 (0.1) 
Epoch 21/256
train loss: 0.0664 - metric: 0.101 - time: 31.6 (1.5) 
valid loss: 0.0511 - metric: 0.087 - time: 31.6 (0.1) 
best
Epoch 22/256
train loss: 0.0632 - metric: 0.099 - time: 33.0 (1.5) 
valid loss: 0.0498 - metric: 0.088 - time: 33.0 (0.1) 
Epoch 23/256
train loss: 0.0620 - metric: 0.098 - time: 34.6 (1.5) 
valid loss: 0.0480 - metric: 0.084 - time: 34.6 (0.1) 
best
Epoch 24/256
train loss: 0.0592 - metric: 0.096 - time: 36.1 (1.5) 
valid loss: 0.0479 - metric: 0.083 - time: 36.1 (0.1) 
best
Epoch 25/256
train loss: 0.0578 - metric: 0.094 - time: 37.5 (1.5) 
valid loss: 0.0466 - metric: 0.084 - time: 37.5 (0.1) 
Epoch 26/256
train loss: 0.0565 - metric: 0.093 - time: 39.0 (1.5) 
valid loss: 0.0459 - metric: 0.085 - time: 39.0 (0.1) 
Epoch 27/256
train loss: 0.0551 - metric: 0.092 - time: 40.5 (1.5) 
valid loss: 0.0452 - metric: 0.081 - time: 40.5 (0.1) 
best
Epoch 28/256
train loss: 0.0542 - metric: 0.091 - time: 42.0 (1.5) 
valid loss: 0.0448 - metric: 0.083 - time: 42.0 (0.1) 
Epoch 29/256
train loss: 0.0534 - metric: 0.090 - time: 43.5 (1.5) 
valid loss: 0.0445 - metric: 0.080 - time: 43.5 (0.1) 
best
Epoch 30/256
train loss: 0.0523 - metric: 0.089 - time: 45.0 (1.5) 
valid loss: 0.0436 - metric: 0.082 - time: 45.0 (0.1) 
Epoch 31/256
train loss: 0.0519 - metric: 0.089 - time: 46.4 (1.5) 
valid loss: 0.0432 - metric: 0.080 - time: 46.4 (0.1) 
best
Epoch 32/256
train loss: 0.0514 - metric: 0.089 - time: 47.9 (1.5) 
valid loss: 0.0431 - metric: 0.080 - time: 47.9 (0.1) 
best
Epoch 33/256
train loss: 0.0502 - metric: 0.087 - time: 49.5 (1.6) 
valid loss: 0.0429 - metric: 0.080 - time: 49.5 (0.1) 
Epoch 34/256
train loss: 0.0503 - metric: 0.088 - time: 51.0 (1.5) 
valid loss: 0.0422 - metric: 0.079 - time: 51.0 (0.1) 
best
Epoch 35/256
train loss: 0.0494 - metric: 0.087 - time: 52.5 (1.5) 
valid loss: 0.0418 - metric: 0.079 - time: 52.5 (0.1) 
best
Epoch 36/256
train loss: 0.0498 - metric: 0.087 - time: 53.9 (1.5) 
valid loss: 0.0420 - metric: 0.078 - time: 53.9 (0.1) 
best
Epoch 37/256
train loss: 0.0488 - metric: 0.086 - time: 55.4 (1.5) 
valid loss: 0.0416 - metric: 0.079 - time: 55.4 (0.1) 
Epoch 38/256
train loss: 0.0479 - metric: 0.086 - time: 56.9 (1.5) 
valid loss: 0.0412 - metric: 0.078 - time: 56.9 (0.1) 
best
Epoch 39/256
train loss: 0.0477 - metric: 0.085 - time: 58.4 (1.6) 
valid loss: 0.0414 - metric: 0.079 - time: 58.4 (0.1) 
Epoch 40/256
train loss: 0.0473 - metric: 0.085 - time: 59.9 (1.4) 
valid loss: 0.0414 - metric: 0.077 - time: 59.9 (0.1) 
best
Epoch 41/256
train loss: 0.0469 - metric: 0.084 - time: 61.4 (1.5) 
valid loss: 0.0409 - metric: 0.079 - time: 61.4 (0.1) 
Epoch 42/256
train loss: 0.0463 - metric: 0.085 - time: 63.0 (1.7) 
valid loss: 0.0403 - metric: 0.077 - time: 63.0 (0.1) 
best
Epoch 43/256
train loss: 0.0469 - metric: 0.084 - time: 64.5 (1.5) 
valid loss: 0.0404 - metric: 0.077 - time: 64.5 (0.1) 
Epoch 44/256
train loss: 0.0463 - metric: 0.084 - time: 66.1 (1.5) 
valid loss: 0.0406 - metric: 0.078 - time: 66.1 (0.1) 
Epoch 45/256
train loss: 0.0457 - metric: 0.084 - time: 67.9 (1.8) 
valid loss: 0.0403 - metric: 0.076 - time: 67.9 (0.1) 
best
Epoch 46/256
train loss: 0.0456 - metric: 0.083 - time: 69.6 (1.7) 
valid loss: 0.0408 - metric: 0.080 - time: 69.6 (0.1) 
Epoch 47/256
train loss: 0.0453 - metric: 0.083 - time: 71.1 (1.5) 
valid loss: 0.0396 - metric: 0.076 - time: 71.1 (0.1) 
Epoch 48/256
train loss: 0.0446 - metric: 0.082 - time: 72.8 (1.7) 
valid loss: 0.0399 - metric: 0.077 - time: 72.8 (0.1) 
Epoch 49/256
train loss: 0.0448 - metric: 0.083 - time: 74.4 (1.6) 
valid loss: 0.0393 - metric: 0.076 - time: 74.4 (0.1) 
Epoch 50/256
train loss: 0.0439 - metric: 0.082 - time: 76.0 (1.6) 
valid loss: 0.0393 - metric: 0.075 - time: 76.0 (0.1) 
best
Epoch 51/256
train loss: 0.0442 - metric: 0.082 - time: 77.4 (1.5) 
valid loss: 0.0394 - metric: 0.077 - time: 77.4 (0.1) 
Epoch 52/256
train loss: 0.0437 - metric: 0.081 - time: 79.0 (1.5) 
valid loss: 0.0393 - metric: 0.075 - time: 79.0 (0.1) 
best
Epoch 53/256
train loss: 0.0436 - metric: 0.081 - time: 80.5 (1.5) 
valid loss: 0.0389 - metric: 0.075 - time: 80.5 (0.1) 
Epoch 54/256
train loss: 0.0430 - metric: 0.081 - time: 82.0 (1.5) 
valid loss: 0.0391 - metric: 0.075 - time: 82.0 (0.1) 
Epoch 55/256
train loss: 0.0434 - metric: 0.081 - time: 83.5 (1.5) 
valid loss: 0.0389 - metric: 0.075 - time: 83.5 (0.1) 
Epoch 56/256
train loss: 0.0427 - metric: 0.080 - time: 85.3 (1.8) 
valid loss: 0.0390 - metric: 0.075 - time: 85.3 (0.1) 
best
Epoch 57/256
train loss: 0.0423 - metric: 0.080 - time: 87.0 (1.7) 
valid loss: 0.0389 - metric: 0.077 - time: 87.0 (0.1) 
Epoch 58/256
train loss: 0.0427 - metric: 0.081 - time: 88.5 (1.5) 
valid loss: 0.0384 - metric: 0.074 - time: 88.5 (0.1) 
best
Epoch 59/256
train loss: 0.0428 - metric: 0.081 - time: 90.0 (1.5) 
valid loss: 0.0382 - metric: 0.075 - time: 90.0 (0.1) 
Epoch 60/256
train loss: 0.0421 - metric: 0.080 - time: 91.5 (1.5) 
valid loss: 0.0388 - metric: 0.075 - time: 91.5 (0.1) 
Epoch 61/256
train loss: 0.0422 - metric: 0.080 - time: 93.3 (1.8) 
valid loss: 0.0386 - metric: 0.074 - time: 93.3 (0.1) 
Epoch 62/256
train loss: 0.0418 - metric: 0.079 - time: 95.2 (1.9) 
valid loss: 0.0382 - metric: 0.074 - time: 95.2 (0.1) 
Epoch 63/256
train loss: 0.0413 - metric: 0.079 - time: 96.9 (1.6) 
valid loss: 0.0380 - metric: 0.074 - time: 96.9 (0.1) 
Epoch 64/256
train loss: 0.0417 - metric: 0.080 - time: 98.4 (1.5) 
valid loss: 0.0379 - metric: 0.074 - time: 98.4 (0.1) 
best
Epoch 65/256
train loss: 0.0410 - metric: 0.079 - time: 99.9 (1.5) 
valid loss: 0.0380 - metric: 0.075 - time: 99.9 (0.1) 
Epoch 66/256
train loss: 0.0410 - metric: 0.079 - time: 101.4 (1.6) 
valid loss: 0.0379 - metric: 0.074 - time: 101.4 (0.1) 
Epoch 67/256
train loss: 0.0410 - metric: 0.079 - time: 103.0 (1.6) 
valid loss: 0.0378 - metric: 0.073 - time: 103.0 (0.1) 
best
Epoch 68/256
train loss: 0.0408 - metric: 0.079 - time: 105.1 (2.1) 
valid loss: 0.0377 - metric: 0.075 - time: 105.1 (0.1) 
Epoch 69/256
train loss: 0.0403 - metric: 0.078 - time: 106.6 (1.5) 
valid loss: 0.0373 - metric: 0.073 - time: 106.6 (0.1) 
best
Epoch 70/256
train loss: 0.0408 - metric: 0.079 - time: 108.1 (1.5) 
valid loss: 0.0375 - metric: 0.074 - time: 108.1 (0.1) 
Epoch 71/256
train loss: 0.0404 - metric: 0.079 - time: 109.6 (1.5) 
valid loss: 0.0373 - metric: 0.074 - time: 109.6 (0.1) 
Epoch 72/256
train loss: 0.0406 - metric: 0.078 - time: 111.1 (1.5) 
valid loss: 0.0375 - metric: 0.074 - time: 111.1 (0.1) 
Epoch 73/256
train loss: 0.0398 - metric: 0.078 - time: 112.6 (1.5) 
valid loss: 0.0373 - metric: 0.073 - time: 112.6 (0.1) 
best
Epoch 74/256
train loss: 0.0399 - metric: 0.078 - time: 114.2 (1.5) 
valid loss: 0.0372 - metric: 0.074 - time: 114.2 (0.1) 
Epoch 75/256
train loss: 0.0397 - metric: 0.078 - time: 115.7 (1.6) 
valid loss: 0.0374 - metric: 0.074 - time: 115.7 (0.1) 
Epoch 76/256
train loss: 0.0398 - metric: 0.078 - time: 117.2 (1.5) 
valid loss: 0.0372 - metric: 0.072 - time: 117.2 (0.1) 
best
Epoch 77/256
train loss: 0.0396 - metric: 0.078 - time: 118.8 (1.5) 
valid loss: 0.0372 - metric: 0.074 - time: 118.8 (0.1) 
Epoch 78/256
train loss: 0.0394 - metric: 0.077 - time: 120.3 (1.6) 
valid loss: 0.0372 - metric: 0.073 - time: 120.3 (0.1) 
Epoch 79/256
train loss: 0.0392 - metric: 0.077 - time: 121.8 (1.5) 
valid loss: 0.0366 - metric: 0.073 - time: 121.8 (0.1) 
Epoch 80/256
train loss: 0.0396 - metric: 0.077 - time: 123.3 (1.5) 
valid loss: 0.0367 - metric: 0.072 - time: 123.3 (0.1) 
best
Epoch 81/256
train loss: 0.0389 - metric: 0.077 - time: 124.8 (1.5) 
valid loss: 0.0369 - metric: 0.074 - time: 124.8 (0.1) 
Epoch 82/256
train loss: 0.0390 - metric: 0.077 - time: 126.3 (1.5) 
valid loss: 0.0368 - metric: 0.072 - time: 126.3 (0.1) 
best
Epoch 83/256
train loss: 0.0388 - metric: 0.076 - time: 127.8 (1.5) 
valid loss: 0.0366 - metric: 0.073 - time: 127.8 (0.1) 
Epoch 84/256
train loss: 0.0382 - metric: 0.076 - time: 129.4 (1.5) 
valid loss: 0.0365 - metric: 0.071 - time: 129.4 (0.1) 
best
Epoch 85/256
train loss: 0.0385 - metric: 0.076 - time: 130.9 (1.5) 
valid loss: 0.0365 - metric: 0.073 - time: 130.9 (0.1) 
Epoch 86/256
train loss: 0.0383 - metric: 0.077 - time: 132.4 (1.5) 
valid loss: 0.0361 - metric: 0.072 - time: 132.4 (0.1) 
Epoch 87/256
train loss: 0.0381 - metric: 0.076 - time: 133.9 (1.5) 
valid loss: 0.0363 - metric: 0.072 - time: 133.9 (0.1) 
Epoch 88/256
train loss: 0.0381 - metric: 0.076 - time: 135.4 (1.5) 
valid loss: 0.0364 - metric: 0.073 - time: 135.4 (0.1) 
Epoch 89/256
train loss: 0.0379 - metric: 0.076 - time: 137.0 (1.6) 
valid loss: 0.0363 - metric: 0.072 - time: 137.0 (0.1) 
Epoch 90/256
train loss: 0.0380 - metric: 0.076 - time: 138.5 (1.5) 
valid loss: 0.0360 - metric: 0.072 - time: 138.5 (0.1) 
Epoch 91/256
train loss: 0.0378 - metric: 0.076 - time: 140.1 (1.6) 
valid loss: 0.0362 - metric: 0.073 - time: 140.1 (0.1) 
Epoch 92/256
train loss: 0.0377 - metric: 0.076 - time: 143.5 (3.4) 
valid loss: 0.0359 - metric: 0.072 - time: 143.5 (0.1) 
Epoch 93/256
train loss: 0.0372 - metric: 0.075 - time: 145.4 (2.0) 
valid loss: 0.0357 - metric: 0.071 - time: 145.4 (0.1) 
best
Epoch 94/256
train loss: 0.0370 - metric: 0.075 - time: 147.0 (1.6) 
valid loss: 0.0357 - metric: 0.071 - time: 147.0 (0.1) 
best
Epoch 95/256
train loss: 0.0368 - metric: 0.075 - time: 148.5 (1.5) 
valid loss: 0.0355 - metric: 0.071 - time: 148.5 (0.1) 
best
Epoch 96/256
train loss: 0.0369 - metric: 0.075 - time: 150.0 (1.5) 
valid loss: 0.0362 - metric: 0.072 - time: 150.0 (0.1) 
Epoch 97/256
train loss: 0.0368 - metric: 0.075 - time: 151.6 (1.6) 
valid loss: 0.0355 - metric: 0.071 - time: 151.6 (0.1) 
Epoch 98/256
train loss: 0.0371 - metric: 0.075 - time: 153.2 (1.6) 
valid loss: 0.0357 - metric: 0.071 - time: 153.2 (0.1) 
best
Epoch 99/256
train loss: 0.0370 - metric: 0.075 - time: 154.8 (1.7) 
valid loss: 0.0360 - metric: 0.072 - time: 154.8 (0.1) 
Epoch 100/256
train loss: 0.0366 - metric: 0.074 - time: 156.4 (1.5) 
valid loss: 0.0354 - metric: 0.071 - time: 156.4 (0.1) 
Epoch 101/256
train loss: 0.0365 - metric: 0.075 - time: 157.9 (1.5) 
valid loss: 0.0354 - metric: 0.071 - time: 157.9 (0.1) 
Epoch 102/256
train loss: 0.0364 - metric: 0.074 - time: 159.5 (1.6) 
valid loss: 0.0358 - metric: 0.072 - time: 159.5 (0.1) 
Epoch 103/256
train loss: 0.0365 - metric: 0.075 - time: 161.0 (1.5) 
valid loss: 0.0357 - metric: 0.070 - time: 161.0 (0.1) 
best
Epoch 104/256
train loss: 0.0360 - metric: 0.074 - time: 162.5 (1.5) 
valid loss: 0.0351 - metric: 0.070 - time: 162.5 (0.1) 
Epoch 105/256
train loss: 0.0360 - metric: 0.074 - time: 164.1 (1.6) 
valid loss: 0.0351 - metric: 0.071 - time: 164.1 (0.1) 
Epoch 106/256
train loss: 0.0361 - metric: 0.074 - time: 165.6 (1.5) 
valid loss: 0.0352 - metric: 0.071 - time: 165.6 (0.1) 
Epoch 107/256
train loss: 0.0358 - metric: 0.074 - time: 167.2 (1.6) 
valid loss: 0.0351 - metric: 0.070 - time: 167.2 (0.1) 
best
Epoch 108/256
train loss: 0.0358 - metric: 0.074 - time: 168.7 (1.5) 
valid loss: 0.0348 - metric: 0.070 - time: 168.7 (0.1) 
Epoch 109/256
train loss: 0.0356 - metric: 0.073 - time: 170.2 (1.5) 
valid loss: 0.0353 - metric: 0.070 - time: 170.2 (0.1) 
Epoch 110/256
train loss: 0.0355 - metric: 0.073 - time: 171.7 (1.6) 
valid loss: 0.0349 - metric: 0.070 - time: 171.7 (0.1) 
best
Epoch 111/256
train loss: 0.0356 - metric: 0.073 - time: 173.3 (1.6) 
valid loss: 0.0351 - metric: 0.071 - time: 173.3 (0.1) 
Epoch 112/256
train loss: 0.0354 - metric: 0.073 - time: 174.8 (1.5) 
valid loss: 0.0346 - metric: 0.070 - time: 174.8 (0.1) 
best
Epoch 113/256
train loss: 0.0354 - metric: 0.073 - time: 176.4 (1.5) 
valid loss: 0.0346 - metric: 0.070 - time: 176.4 (0.1) 
Epoch 114/256
train loss: 0.0353 - metric: 0.073 - time: 177.9 (1.5) 
valid loss: 0.0349 - metric: 0.071 - time: 177.9 (0.1) 
Epoch 115/256
train loss: 0.0353 - metric: 0.073 - time: 179.4 (1.5) 
valid loss: 0.0347 - metric: 0.071 - time: 179.4 (0.1) 
Epoch 116/256
train loss: 0.0350 - metric: 0.073 - time: 181.0 (1.5) 
valid loss: 0.0344 - metric: 0.069 - time: 181.0 (0.1) 
best
Epoch 117/256
train loss: 0.0346 - metric: 0.072 - time: 182.5 (1.5) 
valid loss: 0.0345 - metric: 0.069 - time: 182.5 (0.1) 
Epoch 118/256
train loss: 0.0350 - metric: 0.073 - time: 184.0 (1.5) 
valid loss: 0.0347 - metric: 0.071 - time: 184.0 (0.1) 
Epoch 119/256
train loss: 0.0350 - metric: 0.073 - time: 185.6 (1.5) 
valid loss: 0.0346 - metric: 0.069 - time: 185.6 (0.1) 
best
Epoch 120/256
train loss: 0.0348 - metric: 0.072 - time: 187.1 (1.5) 
valid loss: 0.0347 - metric: 0.070 - time: 187.1 (0.1) 
Epoch 121/256
train loss: 0.0346 - metric: 0.073 - time: 188.6 (1.5) 
valid loss: 0.0345 - metric: 0.070 - time: 188.6 (0.1) 
Epoch 122/256
train loss: 0.0344 - metric: 0.072 - time: 190.1 (1.5) 
valid loss: 0.0342 - metric: 0.069 - time: 190.1 (0.1) 
Epoch 123/256
train loss: 0.0343 - metric: 0.072 - time: 191.7 (1.6) 
valid loss: 0.0347 - metric: 0.070 - time: 191.7 (0.1) 
Epoch 124/256
train loss: 0.0346 - metric: 0.073 - time: 193.3 (1.6) 
valid loss: 0.0349 - metric: 0.071 - time: 193.3 (0.1) 
Epoch 125/256
train loss: 0.0341 - metric: 0.072 - time: 194.8 (1.5) 
valid loss: 0.0340 - metric: 0.069 - time: 194.8 (0.1) 
best
Epoch 126/256
train loss: 0.0342 - metric: 0.072 - time: 196.3 (1.5) 
valid loss: 0.0341 - metric: 0.069 - time: 196.3 (0.1) 
Epoch 127/256
train loss: 0.0341 - metric: 0.072 - time: 197.8 (1.5) 
valid loss: 0.0341 - metric: 0.069 - time: 197.8 (0.1) 
best
Epoch 128/256
train loss: 0.0342 - metric: 0.072 - time: 199.4 (1.5) 
valid loss: 0.0339 - metric: 0.068 - time: 199.4 (0.1) 
best
Epoch 129/256
train loss: 0.0339 - metric: 0.072 - time: 201.0 (1.6) 
valid loss: 0.0342 - metric: 0.070 - time: 201.0 (0.1) 
Epoch 130/256
train loss: 0.0342 - metric: 0.072 - time: 202.5 (1.5) 
valid loss: 0.0340 - metric: 0.069 - time: 202.5 (0.1) 
Epoch 131/256
train loss: 0.0335 - metric: 0.071 - time: 204.1 (1.5) 
valid loss: 0.0341 - metric: 0.069 - time: 204.1 (0.1) 
Epoch 132/256
train loss: 0.0338 - metric: 0.072 - time: 205.6 (1.5) 
valid loss: 0.0340 - metric: 0.068 - time: 205.6 (0.1) 
best
Epoch 133/256
train loss: 0.0336 - metric: 0.072 - time: 207.2 (1.5) 
valid loss: 0.0339 - metric: 0.069 - time: 207.2 (0.1) 
Epoch 134/256
train loss: 0.0333 - metric: 0.071 - time: 208.7 (1.5) 
valid loss: 0.0341 - metric: 0.070 - time: 208.7 (0.1) 
Epoch 135/256
train loss: 0.0336 - metric: 0.072 - time: 210.2 (1.5) 
valid loss: 0.0340 - metric: 0.069 - time: 210.2 (0.1) 
Epoch 136/256
train loss: 0.0335 - metric: 0.071 - time: 211.7 (1.5) 
valid loss: 0.0338 - metric: 0.068 - time: 211.7 (0.1) 
Epoch 137/256
train loss: 0.0331 - metric: 0.071 - time: 213.3 (1.5) 
valid loss: 0.0338 - metric: 0.069 - time: 213.3 (0.1) 
Epoch 138/256
train loss: 0.0330 - metric: 0.071 - time: 214.8 (1.5) 
valid loss: 0.0337 - metric: 0.068 - time: 214.8 (0.1) 
best
Epoch 139/256
train loss: 0.0329 - metric: 0.071 - time: 216.4 (1.6) 
valid loss: 0.0338 - metric: 0.069 - time: 216.4 (0.1) 
Epoch 140/256
train loss: 0.0333 - metric: 0.071 - time: 217.9 (1.6) 
valid loss: 0.0335 - metric: 0.069 - time: 217.9 (0.1) 
Epoch 141/256
train loss: 0.0331 - metric: 0.071 - time: 219.5 (1.5) 
valid loss: 0.0335 - metric: 0.069 - time: 219.5 (0.1) 
Epoch 142/256
train loss: 0.0327 - metric: 0.071 - time: 221.1 (1.6) 
valid loss: 0.0337 - metric: 0.069 - time: 221.1 (0.1) 
Epoch 143/256
train loss: 0.0328 - metric: 0.071 - time: 222.6 (1.5) 
valid loss: 0.0335 - metric: 0.068 - time: 222.6 (0.1) 
best
Epoch 144/256
train loss: 0.0327 - metric: 0.070 - time: 224.1 (1.5) 
valid loss: 0.0336 - metric: 0.069 - time: 224.1 (0.1) 
Epoch 145/256
train loss: 0.0330 - metric: 0.071 - time: 225.6 (1.5) 
valid loss: 0.0338 - metric: 0.070 - time: 225.6 (0.1) 
Epoch 146/256
train loss: 0.0334 - metric: 0.071 - time: 227.2 (1.5) 
valid loss: 0.0341 - metric: 0.070 - time: 227.2 (0.1) 
Epoch 147/256
train loss: 0.0330 - metric: 0.071 - time: 228.7 (1.5) 
valid loss: 0.0335 - metric: 0.068 - time: 228.7 (0.1) 
best
Epoch 148/256
train loss: 0.0325 - metric: 0.070 - time: 230.3 (1.6) 
valid loss: 0.0339 - metric: 0.068 - time: 230.3 (0.1) 
Epoch 149/256
train loss: 0.0326 - metric: 0.070 - time: 231.7 (1.5) 
valid loss: 0.0335 - metric: 0.069 - time: 231.7 (0.1) 
Epoch 150/256
train loss: 0.0327 - metric: 0.070 - time: 233.2 (1.5) 
valid loss: 0.0335 - metric: 0.069 - time: 233.2 (0.1) 
Epoch 151/256
train loss: 0.0323 - metric: 0.070 - time: 234.7 (1.5) 
valid loss: 0.0335 - metric: 0.068 - time: 234.7 (0.1) 
Epoch 152/256
train loss: 0.0328 - metric: 0.071 - time: 236.3 (1.5) 
valid loss: 0.0335 - metric: 0.068 - time: 236.3 (0.1) 
Epoch 153/256
train loss: 0.0323 - metric: 0.070 - time: 237.9 (1.6) 
valid loss: 0.0335 - metric: 0.068 - time: 237.9 (0.1) 
Epoch 154/256
train loss: 0.0323 - metric: 0.070 - time: 239.7 (1.8) 
valid loss: 0.0336 - metric: 0.069 - time: 239.7 (0.1) 
Epoch 155/256
train loss: 0.0322 - metric: 0.070 - time: 241.3 (1.6) 
valid loss: 0.0336 - metric: 0.069 - time: 241.3 (0.1) 
Epoch 156/256
train loss: 0.0325 - metric: 0.070 - time: 242.9 (1.6) 
valid loss: 0.0334 - metric: 0.068 - time: 242.9 (0.1) 
Epoch 157/256
train loss: 0.0324 - metric: 0.070 - time: 244.8 (1.9) 
valid loss: 0.0333 - metric: 0.067 - time: 244.8 (0.1) 
best
Epoch 158/256
train loss: 0.0324 - metric: 0.070 - time: 246.3 (1.5) 
valid loss: 0.0333 - metric: 0.068 - time: 246.3 (0.1) 
Epoch 159/256
train loss: 0.0320 - metric: 0.070 - time: 247.9 (1.5) 
valid loss: 0.0331 - metric: 0.067 - time: 247.9 (0.1) 
best
Epoch 160/256
train loss: 0.0322 - metric: 0.070 - time: 249.3 (1.5) 
valid loss: 0.0331 - metric: 0.068 - time: 249.3 (0.1) 
Epoch 161/256
train loss: 0.0318 - metric: 0.070 - time: 250.8 (1.5) 
valid loss: 0.0336 - metric: 0.069 - time: 250.8 (0.1) 
Epoch 162/256
train loss: 0.0319 - metric: 0.070 - time: 252.4 (1.5) 
valid loss: 0.0337 - metric: 0.070 - time: 252.4 (0.1) 
Epoch 163/256
train loss: 0.0319 - metric: 0.070 - time: 253.9 (1.5) 
valid loss: 0.0332 - metric: 0.068 - time: 253.9 (0.1) 
Epoch 164/256
train loss: 0.0317 - metric: 0.070 - time: 255.4 (1.5) 
valid loss: 0.0331 - metric: 0.068 - time: 255.4 (0.1) 
Epoch 165/256
train loss: 0.0318 - metric: 0.069 - time: 256.9 (1.5) 
valid loss: 0.0330 - metric: 0.068 - time: 256.9 (0.1) 
Epoch 166/256
train loss: 0.0317 - metric: 0.069 - time: 258.5 (1.6) 
valid loss: 0.0333 - metric: 0.068 - time: 258.5 (0.1) 
Epoch 167/256
train loss: 0.0316 - metric: 0.069 - time: 260.1 (1.5) 
valid loss: 0.0331 - metric: 0.068 - time: 260.1 (0.1) 
Epoch 168/256
train loss: 0.0316 - metric: 0.069 - time: 261.6 (1.5) 
valid loss: 0.0333 - metric: 0.068 - time: 261.6 (0.1) 
Epoch 169/256
train loss: 0.0315 - metric: 0.069 - time: 263.2 (1.6) 
valid loss: 0.0333 - metric: 0.068 - time: 263.2 (0.1) 
Epoch 170/256
train loss: 0.0316 - metric: 0.069 - time: 264.7 (1.5) 
valid loss: 0.0334 - metric: 0.069 - time: 264.7 (0.1) 
Epoch 171/256
train loss: 0.0316 - metric: 0.069 - time: 266.2 (1.5) 
valid loss: 0.0334 - metric: 0.069 - time: 266.2 (0.1) 
Epoch 172/256
train loss: 0.0315 - metric: 0.069 - time: 267.8 (1.5) 
valid loss: 0.0333 - metric: 0.069 - time: 267.8 (0.1) 
Epoch 173/256
train loss: 0.0314 - metric: 0.069 - time: 269.3 (1.5) 
valid loss: 0.0332 - metric: 0.068 - time: 269.3 (0.1) 
Epoch 174/256
train loss: 0.0314 - metric: 0.069 - time: 270.9 (1.5) 
valid loss: 0.0332 - metric: 0.068 - time: 270.9 (0.1) 
Epoch 175/256
train loss: 0.0311 - metric: 0.069 - time: 272.4 (1.6) 
valid loss: 0.0332 - metric: 0.068 - time: 272.4 (0.1) 
Epoch 176/256
train loss: 0.0316 - metric: 0.069 - time: 274.0 (1.5) 
valid loss: 0.0330 - metric: 0.067 - time: 274.0 (0.1) 
best
Epoch 177/256
train loss: 0.0314 - metric: 0.069 - time: 275.5 (1.5) 
valid loss: 0.0331 - metric: 0.068 - time: 275.5 (0.1) 
Epoch 178/256
train loss: 0.0313 - metric: 0.069 - time: 277.1 (1.6) 
valid loss: 0.0330 - metric: 0.068 - time: 277.1 (0.1) 
Epoch 179/256
train loss: 0.0315 - metric: 0.069 - time: 278.6 (1.5) 
valid loss: 0.0330 - metric: 0.068 - time: 278.6 (0.1) 
Epoch 180/256
train loss: 0.0312 - metric: 0.069 - time: 280.1 (1.5) 
valid loss: 0.0329 - metric: 0.068 - time: 280.1 (0.1) 
Epoch 181/256
train loss: 0.0309 - metric: 0.069 - time: 281.6 (1.5) 
valid loss: 0.0330 - metric: 0.067 - time: 281.6 (0.1) 
Epoch 182/256
train loss: 0.0311 - metric: 0.069 - time: 283.2 (1.5) 
valid loss: 0.0330 - metric: 0.068 - time: 283.2 (0.1) 
Epoch 183/256
train loss: 0.0309 - metric: 0.069 - time: 284.7 (1.5) 
valid loss: 0.0330 - metric: 0.068 - time: 284.7 (0.1) 
Epoch 184/256
train loss: 0.0309 - metric: 0.069 - time: 286.2 (1.5) 
valid loss: 0.0330 - metric: 0.068 - time: 286.2 (0.1) 
Epoch 185/256
train loss: 0.0314 - metric: 0.069 - time: 287.8 (1.6) 
valid loss: 0.0329 - metric: 0.068 - time: 287.8 (0.1) 
Epoch 186/256
train loss: 0.0311 - metric: 0.069 - time: 289.3 (1.5) 
valid loss: 0.0329 - metric: 0.067 - time: 289.3 (0.1) 
Epoch 187/256
train loss: 0.0309 - metric: 0.068 - time: 290.9 (1.5) 
valid loss: 0.0329 - metric: 0.068 - time: 290.9 (0.1) 
Epoch 188/256
train loss: 0.0308 - metric: 0.069 - time: 292.4 (1.5) 
valid loss: 0.0330 - metric: 0.067 - time: 292.4 (0.1) 
Epoch 189/256
train loss: 0.0308 - metric: 0.069 - time: 293.9 (1.5) 
valid loss: 0.0329 - metric: 0.067 - time: 293.9 (0.1) 
Epoch 190/256
train loss: 0.0311 - metric: 0.069 - time: 295.4 (1.5) 
valid loss: 0.0330 - metric: 0.068 - time: 295.4 (0.1) 
Epoch 191/256
train loss: 0.0312 - metric: 0.069 - time: 296.9 (1.5) 
valid loss: 0.0330 - metric: 0.068 - time: 296.9 (0.1) 
Epoch 192/256
train loss: 0.0306 - metric: 0.068 - time: 298.5 (1.5) 
valid loss: 0.0330 - metric: 0.068 - time: 298.5 (0.1) 
Epoch 193/256
train loss: 0.0311 - metric: 0.069 - time: 300.0 (1.6) 
valid loss: 0.0329 - metric: 0.067 - time: 300.0 (0.1) 
Epoch 194/256
train loss: 0.0311 - metric: 0.069 - time: 301.6 (1.6) 
valid loss: 0.0329 - metric: 0.067 - time: 301.6 (0.1) 
best
Epoch 195/256
train loss: 0.0309 - metric: 0.069 - time: 303.2 (1.5) 
valid loss: 0.0329 - metric: 0.067 - time: 303.2 (0.1) 
Epoch 196/256
train loss: 0.0310 - metric: 0.069 - time: 304.7 (1.5) 
valid loss: 0.0331 - metric: 0.068 - time: 304.7 (0.1) 
Epoch 197/256
train loss: 0.0307 - metric: 0.068 - time: 306.2 (1.5) 
valid loss: 0.0330 - metric: 0.068 - time: 306.2 (0.1) 
Epoch 198/256
train loss: 0.0309 - metric: 0.069 - time: 307.8 (1.6) 
valid loss: 0.0330 - metric: 0.067 - time: 307.8 (0.1) 
Epoch 199/256
train loss: 0.0310 - metric: 0.068 - time: 309.4 (1.5) 
valid loss: 0.0330 - metric: 0.068 - time: 309.4 (0.1) 
Epoch 200/256
train loss: 0.0305 - metric: 0.068 - time: 310.8 (1.5) 
valid loss: 0.0329 - metric: 0.067 - time: 310.8 (0.1) 
Epoch 201/256
train loss: 0.0310 - metric: 0.069 - time: 312.4 (1.5) 
valid loss: 0.0329 - metric: 0.067 - time: 312.4 (0.1) 
Epoch 202/256
train loss: 0.0307 - metric: 0.068 - time: 313.9 (1.5) 
valid loss: 0.0331 - metric: 0.068 - time: 313.9 (0.1) 
Epoch 203/256
train loss: 0.0307 - metric: 0.069 - time: 315.5 (1.6) 
valid loss: 0.0328 - metric: 0.067 - time: 315.5 (0.1) 
Epoch 204/256
train loss: 0.0308 - metric: 0.068 - time: 317.0 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 317.0 (0.1) 
Epoch 205/256
train loss: 0.0305 - metric: 0.068 - time: 318.5 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 318.5 (0.1) 
Epoch 206/256
train loss: 0.0306 - metric: 0.068 - time: 320.1 (1.5) 
valid loss: 0.0329 - metric: 0.067 - time: 320.1 (0.1) 
Epoch 207/256
train loss: 0.0302 - metric: 0.068 - time: 321.7 (1.6) 
valid loss: 0.0328 - metric: 0.067 - time: 321.7 (0.1) 
Epoch 208/256
train loss: 0.0305 - metric: 0.068 - time: 323.2 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 323.2 (0.1) 
Epoch 209/256
train loss: 0.0303 - metric: 0.068 - time: 324.8 (1.6) 
valid loss: 0.0328 - metric: 0.067 - time: 324.8 (0.1) 
Epoch 210/256
train loss: 0.0305 - metric: 0.068 - time: 326.3 (1.5) 
valid loss: 0.0329 - metric: 0.068 - time: 326.3 (0.1) 
Epoch 211/256
train loss: 0.0307 - metric: 0.069 - time: 327.8 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 327.8 (0.1) 
Epoch 212/256
train loss: 0.0305 - metric: 0.068 - time: 329.4 (1.5) 
valid loss: 0.0329 - metric: 0.068 - time: 329.4 (0.1) 
Epoch 213/256
train loss: 0.0304 - metric: 0.068 - time: 330.9 (1.6) 
valid loss: 0.0328 - metric: 0.067 - time: 330.9 (0.1) 
Epoch 214/256
train loss: 0.0303 - metric: 0.068 - time: 332.5 (1.5) 
valid loss: 0.0329 - metric: 0.068 - time: 332.5 (0.1) 
Epoch 215/256
train loss: 0.0308 - metric: 0.069 - time: 334.0 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 334.0 (0.1) 
Epoch 216/256
train loss: 0.0307 - metric: 0.068 - time: 335.6 (1.6) 
valid loss: 0.0328 - metric: 0.067 - time: 335.6 (0.1) 
Epoch 217/256
train loss: 0.0304 - metric: 0.068 - time: 337.1 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 337.1 (0.1) 
Epoch 218/256
train loss: 0.0304 - metric: 0.068 - time: 338.7 (1.6) 
valid loss: 0.0328 - metric: 0.067 - time: 338.7 (0.1) 
Epoch 219/256
train loss: 0.0304 - metric: 0.068 - time: 340.2 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 340.2 (0.1) 
Epoch 220/256
train loss: 0.0303 - metric: 0.068 - time: 341.7 (1.5) 
valid loss: 0.0329 - metric: 0.067 - time: 341.7 (0.1) 
Epoch 221/256
train loss: 0.0307 - metric: 0.068 - time: 343.3 (1.6) 
valid loss: 0.0328 - metric: 0.067 - time: 343.3 (0.1) 
Epoch 222/256
train loss: 0.0302 - metric: 0.068 - time: 344.8 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 344.8 (0.1) 
Epoch 223/256
train loss: 0.0305 - metric: 0.068 - time: 346.4 (1.6) 
valid loss: 0.0328 - metric: 0.067 - time: 346.4 (0.1) 
Epoch 224/256
train loss: 0.0306 - metric: 0.068 - time: 348.0 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 348.0 (0.1) 
Epoch 225/256
train loss: 0.0305 - metric: 0.068 - time: 349.5 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 349.5 (0.1) 
Epoch 226/256
train loss: 0.0304 - metric: 0.068 - time: 351.0 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 351.0 (0.1) 
Epoch 227/256
train loss: 0.0305 - metric: 0.068 - time: 352.6 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 352.6 (0.1) 
Epoch 228/256
train loss: 0.0306 - metric: 0.068 - time: 354.1 (1.6) 
valid loss: 0.0327 - metric: 0.067 - time: 354.1 (0.1) 
Epoch 229/256
train loss: 0.0305 - metric: 0.068 - time: 355.6 (1.5) 
valid loss: 0.0327 - metric: 0.067 - time: 355.6 (0.1) 
Epoch 230/256
train loss: 0.0303 - metric: 0.068 - time: 357.2 (1.6) 
valid loss: 0.0328 - metric: 0.067 - time: 357.2 (0.1) 
Epoch 231/256
train loss: 0.0309 - metric: 0.069 - time: 358.7 (1.6) 
valid loss: 0.0328 - metric: 0.067 - time: 358.7 (0.1) 
Epoch 232/256
train loss: 0.0307 - metric: 0.068 - time: 360.3 (1.6) 
valid loss: 0.0328 - metric: 0.067 - time: 360.3 (0.1) 
Epoch 233/256
train loss: 0.0304 - metric: 0.068 - time: 361.8 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 361.8 (0.1) 
Epoch 234/256
train loss: 0.0307 - metric: 0.069 - time: 363.4 (1.6) 
valid loss: 0.0328 - metric: 0.067 - time: 363.4 (0.1) 
Epoch 235/256
train loss: 0.0305 - metric: 0.068 - time: 364.9 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 364.9 (0.1) 
Epoch 236/256
train loss: 0.0305 - metric: 0.068 - time: 366.5 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 366.5 (0.1) 
Epoch 237/256
train loss: 0.0303 - metric: 0.068 - time: 368.0 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 368.0 (0.1) 
Epoch 238/256
train loss: 0.0302 - metric: 0.068 - time: 369.6 (1.5) 
valid loss: 0.0328 - metric: 0.067 - time: 369.6 (0.1) 
Epoch 239/256
train loss: 0.0304 - metric: 0.068 - time: 371.1 (1.5) 
valid loss: 0.0327 - metric: 0.067 - time: 371.1 (0.1) 
Epoch 240/256
train loss: 0.0301 - metric: 0.068 - time: 372.6 (1.5) 
valid loss: 0.0327 - metric: 0.067 - time: 372.6 (0.1) 
Epoch 241/256
train loss: 0.0306 - metric: 0.068 - time: 374.1 (1.5) 
valid loss: 0.0327 - metric: 0.067 - time: 374.1 (0.1) 
Epoch 242/256
train loss: 0.0305 - metric: 0.068 - time: 375.7 (1.5) 
valid loss: 0.0327 - metric: 0.067 - time: 375.7 (0.1) 
Epoch 243/256
train loss: 0.0303 - metric: 0.068 - time: 377.2 (1.5) 
valid loss: 0.0327 - metric: 0.067 - time: 377.2 (0.1) 
Epoch 244/256
train loss: 0.0306 - metric: 0.068 - time: 378.8 (1.6) 
valid loss: 0.0327 - metric: 0.067 - time: 378.8 (0.1) 
Epoch 245/256
train loss: 0.0303 - metric: 0.068 - time: 380.4 (1.5) 
valid loss: 0.0327 - metric: 0.067 - time: 380.4 (0.1) 
Epoch 246/256
train loss: 0.0301 - metric: 0.068 - time: 381.9 (1.6) 
valid loss: 0.0327 - metric: 0.067 - time: 381.9 (0.1) 
Epoch 247/256
train loss: 0.0306 - metric: 0.068 - time: 383.5 (1.6) 
valid loss: 0.0327 - metric: 0.067 - time: 383.5 (0.1) 
Epoch 248/256
train loss: 0.0306 - metric: 0.068 - time: 385.0 (1.5) 
valid loss: 0.0327 - metric: 0.067 - time: 385.0 (0.1) 
Epoch 249/256
train loss: 0.0307 - metric: 0.069 - time: 386.6 (1.6) 
valid loss: 0.0327 - metric: 0.067 - time: 386.6 (0.1) 
Epoch 250/256
train loss: 0.0306 - metric: 0.068 - time: 388.2 (1.6) 
valid loss: 0.0327 - metric: 0.067 - time: 388.2 (0.1) 
Epoch 251/256
train loss: 0.0305 - metric: 0.068 - time: 389.8 (1.6) 
valid loss: 0.0327 - metric: 0.067 - time: 389.8 (0.1) 
Epoch 252/256
train loss: 0.0304 - metric: 0.068 - time: 391.3 (1.5) 
valid loss: 0.0327 - metric: 0.067 - time: 391.3 (0.1) 
Epoch 253/256
train loss: 0.0304 - metric: 0.068 - time: 392.9 (1.6) 
valid loss: 0.0327 - metric: 0.067 - time: 392.9 (0.1) 
Epoch 254/256
train loss: 0.0305 - metric: 0.068 - time: 394.5 (1.5) 
valid loss: 0.0327 - metric: 0.067 - time: 394.5 (0.1) 
Epoch 255/256
train loss: 0.0302 - metric: 0.068 - time: 396.0 (1.5) 
valid loss: 0.0327 - metric: 0.067 - time: 396.0 (0.1) 
Epoch 256/256
train loss: 0.0302 - metric: 0.068 - time: 397.5 (1.5) 
valid loss: 0.0327 - metric: 0.067 - time: 397.5 (0.1) 

best epoch: 194 - metric: 0.06696902460560908 - loss: 0.03287146608092949


"""