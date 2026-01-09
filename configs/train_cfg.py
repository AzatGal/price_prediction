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

cfg.loss = 'L1Loss'  # CrossEntropyLoss SmoothL1Loss KLDivLoss L1Loss
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
load_pretrained
cpu
Epoch 1/125
train loss: 1.1705 - metric: 1.079 - time: 3.3 (3.3) 
valid loss: 0.9146 - metric: 0.587 - time: 3.3 (0.2) 
best
Epoch 2/125
train loss: 0.7716 - metric: 0.488 - time: 6.5 (3.2) 
valid loss: 0.8043 - metric: 0.432 - time: 6.5 (0.2) 
best
Epoch 3/125
train loss: 0.6313 - metric: 0.380 - time: 9.7 (3.2) 
valid loss: 0.5158 - metric: 0.264 - time: 9.7 (0.2) 
best
Epoch 4/125
train loss: 0.4299 - metric: 0.236 - time: 13.0 (3.3) 
valid loss: 0.4212 - metric: 0.162 - time: 13.0 (0.2) 
best
Epoch 5/125
train loss: 0.3240 - metric: 0.175 - time: 16.2 (3.3) 
valid loss: 0.3939 - metric: 0.135 - time: 16.2 (0.2) 
best
Epoch 6/125
train loss: 0.2855 - metric: 0.155 - time: 19.5 (3.3) 
valid loss: 0.3983 - metric: 0.120 - time: 19.5 (0.2) 
best
Epoch 7/125
train loss: 0.2538 - metric: 0.137 - time: 22.8 (3.2) 
valid loss: 0.3783 - metric: 0.110 - time: 22.8 (0.2) 
best
Epoch 8/125
train loss: 0.2349 - metric: 0.127 - time: 26.1 (3.3) 
valid loss: 0.3425 - metric: 0.099 - time: 26.1 (0.2) 
best
Epoch 9/125
train loss: 0.2173 - metric: 0.118 - time: 29.3 (3.2) 
valid loss: 0.3401 - metric: 0.098 - time: 29.3 (0.2) 
best
Epoch 10/125
train loss: 0.2024 - metric: 0.109 - time: 32.6 (3.3) 
valid loss: 0.2985 - metric: 0.089 - time: 32.6 (0.2) 
best
Epoch 11/125
train loss: 0.1933 - metric: 0.104 - time: 35.9 (3.3) 
valid loss: 0.2952 - metric: 0.091 - time: 35.9 (0.2) 
Epoch 12/125
train loss: 0.1873 - metric: 0.101 - time: 39.1 (3.2) 
valid loss: 0.2860 - metric: 0.094 - time: 39.1 (0.2) 
Epoch 13/125
train loss: 0.1810 - metric: 0.098 - time: 42.4 (3.3) 
valid loss: 0.2693 - metric: 0.084 - time: 42.4 (0.2) 
best
Epoch 14/125
train loss: 0.1816 - metric: 0.099 - time: 45.7 (3.4) 
valid loss: 0.2931 - metric: 0.089 - time: 45.7 (0.2) 
Epoch 15/125
train loss: 0.1745 - metric: 0.094 - time: 49.0 (3.3) 
valid loss: 0.2541 - metric: 0.081 - time: 49.0 (0.2) 
best
Epoch 16/125
train loss: 0.1682 - metric: 0.091 - time: 52.3 (3.3) 
valid loss: 0.2822 - metric: 0.085 - time: 52.3 (0.2) 
Epoch 17/125
train loss: 0.1641 - metric: 0.089 - time: 55.6 (3.3) 
valid loss: 0.2587 - metric: 0.083 - time: 55.6 (0.2) 
Epoch 18/125
train loss: 0.1678 - metric: 0.091 - time: 58.9 (3.3) 
valid loss: 0.2666 - metric: 0.082 - time: 58.9 (0.2) 
Epoch 19/125
train loss: 0.1602 - metric: 0.087 - time: 62.1 (3.3) 
valid loss: 0.2622 - metric: 0.081 - time: 62.1 (0.2) 
best
Epoch 20/125
train loss: 0.1537 - metric: 0.083 - time: 65.5 (3.3) 
valid loss: 0.2424 - metric: 0.074 - time: 65.5 (0.2) 
best
Epoch 21/125
train loss: 0.1531 - metric: 0.083 - time: 68.7 (3.3) 
valid loss: 0.2488 - metric: 0.074 - time: 68.7 (0.2) 
Epoch 22/125
train loss: 0.1594 - metric: 0.087 - time: 72.0 (3.3) 
valid loss: 0.2456 - metric: 0.074 - time: 72.0 (0.2) 
Epoch 23/125
train loss: 0.1575 - metric: 0.085 - time: 75.3 (3.3) 
valid loss: 0.2764 - metric: 0.087 - time: 75.3 (0.2) 
Epoch 24/125
train loss: 0.1505 - metric: 0.082 - time: 78.6 (3.3) 
valid loss: 0.2675 - metric: 0.079 - time: 78.6 (0.2) 
Epoch 25/125
train loss: 0.1462 - metric: 0.079 - time: 81.9 (3.3) 
valid loss: 0.2488 - metric: 0.074 - time: 81.9 (0.2) 
best
Epoch 26/125
train loss: 0.1451 - metric: 0.079 - time: 85.2 (3.3) 
valid loss: 0.2511 - metric: 0.072 - time: 85.2 (0.2) 
best
Epoch 27/125
train loss: 0.1472 - metric: 0.080 - time: 88.5 (3.3) 
valid loss: 0.2890 - metric: 0.087 - time: 88.5 (0.2) 
Epoch 28/125
train loss: 0.1469 - metric: 0.080 - time: 91.7 (3.3) 
valid loss: 0.2633 - metric: 0.075 - time: 91.7 (0.2) 
Epoch 29/125
train loss: 0.1419 - metric: 0.077 - time: 95.0 (3.3) 
valid loss: 0.2493 - metric: 0.073 - time: 95.0 (0.2) 
Epoch 30/125
train loss: 0.1391 - metric: 0.076 - time: 98.3 (3.3) 
valid loss: 0.2521 - metric: 0.074 - time: 98.3 (0.2) 
Epoch 31/125
train loss: 0.1383 - metric: 0.075 - time: 101.6 (3.4) 
valid loss: 0.2428 - metric: 0.068 - time: 101.6 (0.2) 
best
Epoch 32/125
train loss: 0.1366 - metric: 0.074 - time: 105.0 (3.3) 
valid loss: 0.2555 - metric: 0.071 - time: 105.0 (0.2) 
Epoch 33/125
train loss: 0.1364 - metric: 0.074 - time: 108.3 (3.3) 
valid loss: 0.2467 - metric: 0.069 - time: 108.3 (0.2) 
Epoch 34/125
train loss: 0.1343 - metric: 0.073 - time: 111.6 (3.3) 
valid loss: 0.2345 - metric: 0.066 - time: 111.6 (0.2) 
best
Epoch 35/125
train loss: 0.1330 - metric: 0.072 - time: 114.9 (3.3) 
valid loss: 0.2383 - metric: 0.068 - time: 114.9 (0.2) 
Epoch 36/125
train loss: 0.1319 - metric: 0.072 - time: 118.2 (3.3) 
valid loss: 0.2492 - metric: 0.070 - time: 118.2 (0.2) 
Epoch 37/125
train loss: 0.1305 - metric: 0.071 - time: 121.5 (3.3) 
valid loss: 0.2489 - metric: 0.069 - time: 121.5 (0.2) 
Epoch 38/125
train loss: 0.1306 - metric: 0.071 - time: 124.8 (3.3) 
valid loss: 0.2520 - metric: 0.068 - time: 124.8 (0.2) 
Epoch 39/125
train loss: 0.1296 - metric: 0.070 - time: 128.1 (3.4) 
valid loss: 0.2461 - metric: 0.068 - time: 128.1 (0.2) 
Epoch 40/125
train loss: 0.1271 - metric: 0.069 - time: 131.6 (3.5) 
valid loss: 0.2479 - metric: 0.068 - time: 131.6 (0.2) 
Epoch 41/125
train loss: 0.1278 - metric: 0.070 - time: 134.9 (3.3) 
valid loss: 0.2501 - metric: 0.068 - time: 134.9 (0.2) 
Epoch 42/125
train loss: 0.1257 - metric: 0.068 - time: 138.2 (3.3) 
valid loss: 0.2401 - metric: 0.067 - time: 138.2 (0.2) 
Epoch 43/125
train loss: 0.1245 - metric: 0.068 - time: 141.5 (3.3) 
valid loss: 0.2386 - metric: 0.067 - time: 141.5 (0.2) 
Epoch 44/125
train loss: 0.1255 - metric: 0.068 - time: 144.8 (3.3) 
valid loss: 0.2567 - metric: 0.073 - time: 144.8 (0.2) 
Epoch 45/125
train loss: 0.1257 - metric: 0.068 - time: 148.1 (3.3) 
valid loss: 0.2423 - metric: 0.066 - time: 148.1 (0.2) 
best
Epoch 46/125
train loss: 0.1230 - metric: 0.067 - time: 151.4 (3.3) 
valid loss: 0.2480 - metric: 0.066 - time: 151.4 (0.2) 
Epoch 47/125
train loss: 0.1223 - metric: 0.066 - time: 154.7 (3.3) 
valid loss: 0.2386 - metric: 0.065 - time: 154.7 (0.2) 
best
Epoch 48/125
train loss: 0.1214 - metric: 0.066 - time: 158.0 (3.3) 
valid loss: 0.2469 - metric: 0.067 - time: 158.0 (0.2) 
Epoch 49/125
train loss: 0.1208 - metric: 0.066 - time: 161.4 (3.3) 
valid loss: 0.2563 - metric: 0.069 - time: 161.4 (0.2) 
Epoch 50/125
train loss: 0.1212 - metric: 0.066 - time: 164.6 (3.3) 
valid loss: 0.2435 - metric: 0.065 - time: 164.6 (0.2) 
Epoch 51/125
train loss: 0.1203 - metric: 0.065 - time: 168.0 (3.3) 
valid loss: 0.2246 - metric: 0.063 - time: 168.0 (0.2) 
best
Epoch 52/125
train loss: 0.1196 - metric: 0.065 - time: 171.2 (3.3) 
valid loss: 0.2506 - metric: 0.067 - time: 171.2 (0.2) 
Epoch 53/125
train loss: 0.1192 - metric: 0.065 - time: 174.5 (3.3) 
valid loss: 0.2411 - metric: 0.065 - time: 174.5 (0.2) 
Epoch 54/125
train loss: 0.1181 - metric: 0.064 - time: 177.9 (3.3) 
valid loss: 0.2440 - metric: 0.065 - time: 177.9 (0.2) 
Epoch 55/125
train loss: 0.1176 - metric: 0.064 - time: 181.2 (3.3) 
valid loss: 0.2487 - metric: 0.066 - time: 181.2 (0.2) 
Epoch 56/125
train loss: 0.1165 - metric: 0.063 - time: 184.4 (3.3) 
valid loss: 0.2282 - metric: 0.061 - time: 184.4 (0.2) 
best
Epoch 57/125
train loss: 0.1163 - metric: 0.063 - time: 187.8 (3.4) 
valid loss: 0.2441 - metric: 0.064 - time: 187.8 (0.2) 
Epoch 58/125
train loss: 0.1162 - metric: 0.063 - time: 191.1 (3.3) 
valid loss: 0.2398 - metric: 0.063 - time: 191.1 (0.2) 
Epoch 59/125
train loss: 0.1156 - metric: 0.063 - time: 194.4 (3.3) 
valid loss: 0.2357 - metric: 0.062 - time: 194.4 (0.2) 
Epoch 60/125
train loss: 0.1142 - metric: 0.062 - time: 197.7 (3.3) 
valid loss: 0.2365 - metric: 0.062 - time: 197.7 (0.2) 
Epoch 61/125
train loss: 0.1148 - metric: 0.062 - time: 201.0 (3.3) 
valid loss: 0.2486 - metric: 0.064 - time: 201.0 (0.2) 
Epoch 62/125
train loss: 0.1140 - metric: 0.062 - time: 204.3 (3.3) 
valid loss: 0.2469 - metric: 0.064 - time: 204.3 (0.2) 
Epoch 63/125
train loss: 0.1135 - metric: 0.062 - time: 207.6 (3.3) 
valid loss: 0.2432 - metric: 0.063 - time: 207.6 (0.2) 
Epoch 64/125
train loss: 0.1141 - metric: 0.062 - time: 210.9 (3.3) 
valid loss: 0.2369 - metric: 0.061 - time: 210.9 (0.2) 
best
Epoch 65/125
train loss: 0.1130 - metric: 0.061 - time: 214.2 (3.3) 
valid loss: 0.2457 - metric: 0.063 - time: 214.2 (0.2) 
Epoch 66/125
train loss: 0.1127 - metric: 0.061 - time: 217.5 (3.3) 
valid loss: 0.2427 - metric: 0.063 - time: 217.5 (0.2) 
Epoch 67/125
train loss: 0.1131 - metric: 0.062 - time: 220.8 (3.3) 
valid loss: 0.2359 - metric: 0.061 - time: 220.8 (0.2) 
best
Epoch 68/125
train loss: 0.1118 - metric: 0.061 - time: 224.1 (3.3) 
valid loss: 0.2280 - metric: 0.060 - time: 224.1 (0.2) 
best
Epoch 69/125
train loss: 0.1119 - metric: 0.061 - time: 227.5 (3.3) 
valid loss: 0.2414 - metric: 0.063 - time: 227.5 (0.2) 
Epoch 70/125
train loss: 0.1113 - metric: 0.060 - time: 230.8 (3.3) 
valid loss: 0.2350 - metric: 0.061 - time: 230.8 (0.2) 
Epoch 71/125
train loss: 0.1114 - metric: 0.061 - time: 234.1 (3.3) 
valid loss: 0.2345 - metric: 0.061 - time: 234.1 (0.2) 
Epoch 72/125
train loss: 0.1110 - metric: 0.060 - time: 237.4 (3.3) 
valid loss: 0.2391 - metric: 0.062 - time: 237.4 (0.2) 
Epoch 73/125
train loss: 0.1100 - metric: 0.060 - time: 240.7 (3.3) 
valid loss: 0.2482 - metric: 0.064 - time: 240.7 (0.2) 
Epoch 74/125
train loss: 0.1102 - metric: 0.060 - time: 244.1 (3.4) 
valid loss: 0.2349 - metric: 0.061 - time: 244.1 (0.2) 
Epoch 75/125
train loss: 0.1091 - metric: 0.059 - time: 247.4 (3.3) 
valid loss: 0.2453 - metric: 0.063 - time: 247.4 (0.2) 
Epoch 76/125
train loss: 0.1095 - metric: 0.060 - time: 250.7 (3.3) 
valid loss: 0.2472 - metric: 0.064 - time: 250.7 (0.2) 
Epoch 77/125
train loss: 0.1088 - metric: 0.059 - time: 254.1 (3.3) 
valid loss: 0.2340 - metric: 0.061 - time: 254.1 (0.2) 
Epoch 78/125
train loss: 0.1088 - metric: 0.059 - time: 257.4 (3.3) 
valid loss: 0.2419 - metric: 0.063 - time: 257.4 (0.2) 
Epoch 79/125
train loss: 0.1088 - metric: 0.059 - time: 260.7 (3.3) 
valid loss: 0.2423 - metric: 0.062 - time: 260.7 (0.2) 
Epoch 80/125
train loss: 0.1083 - metric: 0.059 - time: 264.0 (3.3) 
valid loss: 0.2351 - metric: 0.061 - time: 264.0 (0.2) 
Epoch 81/125
train loss: 0.1078 - metric: 0.059 - time: 267.3 (3.3) 
valid loss: 0.2336 - metric: 0.060 - time: 267.3 (0.2) 
Epoch 82/125
train loss: 0.1077 - metric: 0.059 - time: 270.6 (3.3) 
valid loss: 0.2474 - metric: 0.062 - time: 270.6 (0.2) 
Epoch 83/125
train loss: 0.1073 - metric: 0.058 - time: 274.1 (3.5) 
valid loss: 0.2503 - metric: 0.063 - time: 274.1 (0.2) 
Epoch 84/125
train loss: 0.1076 - metric: 0.058 - time: 277.5 (3.4) 
valid loss: 0.2365 - metric: 0.060 - time: 277.5 (0.2) 
Epoch 85/125
train loss: 0.1065 - metric: 0.058 - time: 280.8 (3.3) 
valid loss: 0.2436 - metric: 0.061 - time: 280.8 (0.2) 
Epoch 86/125
train loss: 0.1067 - metric: 0.058 - time: 284.1 (3.3) 
valid loss: 0.2393 - metric: 0.060 - time: 284.1 (0.2) 
Epoch 87/125
train loss: 0.1068 - metric: 0.058 - time: 287.4 (3.3) 
valid loss: 0.2474 - metric: 0.062 - time: 287.4 (0.2) 
Epoch 88/125
train loss: 0.1067 - metric: 0.058 - time: 290.6 (3.3) 
valid loss: 0.2438 - metric: 0.062 - time: 290.6 (0.2) 
Epoch 89/125
train loss: 0.1056 - metric: 0.057 - time: 293.9 (3.3) 
valid loss: 0.2416 - metric: 0.062 - time: 293.9 (0.2) 
Epoch 90/125
train loss: 0.1055 - metric: 0.057 - time: 297.2 (3.3) 
valid loss: 0.2343 - metric: 0.060 - time: 297.2 (0.2) 
Epoch 91/125
train loss: 0.1052 - metric: 0.057 - time: 300.5 (3.3) 
valid loss: 0.2437 - metric: 0.062 - time: 300.5 (0.2) 
Epoch 92/125
train loss: 0.1050 - metric: 0.057 - time: 303.8 (3.3) 
valid loss: 0.2343 - metric: 0.060 - time: 303.8 (0.2) 
Epoch 93/125
train loss: 0.1057 - metric: 0.058 - time: 307.1 (3.3) 
valid loss: 0.2432 - metric: 0.062 - time: 307.1 (0.2) 
Epoch 94/125
train loss: 0.1057 - metric: 0.057 - time: 310.4 (3.3) 
valid loss: 0.2394 - metric: 0.061 - time: 310.4 (0.2) 
Epoch 95/125
train loss: 0.1044 - metric: 0.057 - time: 313.7 (3.3) 
valid loss: 0.2399 - metric: 0.061 - time: 313.7 (0.2) 
Epoch 96/125
train loss: 0.1051 - metric: 0.057 - time: 317.0 (3.3) 
valid loss: 0.2354 - metric: 0.060 - time: 317.0 (0.2) 
Epoch 97/125
train loss: 0.1044 - metric: 0.057 - time: 320.4 (3.3) 
valid loss: 0.2475 - metric: 0.062 - time: 320.4 (0.2) 
Epoch 98/125
train loss: 0.1044 - metric: 0.057 - time: 323.7 (3.3) 
valid loss: 0.2369 - metric: 0.059 - time: 323.7 (0.2) 
best
Epoch 99/125
train loss: 0.1052 - metric: 0.057 - time: 326.9 (3.3) 
valid loss: 0.2388 - metric: 0.060 - time: 326.9 (0.2) 
Epoch 100/125
train loss: 0.1039 - metric: 0.056 - time: 330.3 (3.4) 
valid loss: 0.2428 - metric: 0.060 - time: 330.3 (0.2) 
Epoch 101/125
train loss: 0.1037 - metric: 0.056 - time: 333.6 (3.3) 
valid loss: 0.2506 - metric: 0.062 - time: 333.6 (0.2) 
Epoch 102/125
train loss: 0.1036 - metric: 0.056 - time: 336.9 (3.3) 
valid loss: 0.2456 - metric: 0.062 - time: 336.9 (0.2) 
Epoch 103/125
train loss: 0.1033 - metric: 0.056 - time: 340.2 (3.3) 
valid loss: 0.2443 - metric: 0.061 - time: 340.2 (0.2) 
Epoch 104/125
train loss: 0.1029 - metric: 0.056 - time: 343.5 (3.3) 
valid loss: 0.2385 - metric: 0.060 - time: 343.5 (0.2) 
Epoch 105/125
train loss: 0.1023 - metric: 0.056 - time: 346.8 (3.3) 
valid loss: 0.2375 - metric: 0.060 - time: 346.8 (0.2) 
Epoch 106/125
train loss: 0.1028 - metric: 0.056 - time: 350.1 (3.3) 
valid loss: 0.2424 - metric: 0.060 - time: 350.1 (0.2) 
Epoch 107/125
train loss: 0.1032 - metric: 0.056 - time: 353.4 (3.3) 
valid loss: 0.2459 - metric: 0.060 - time: 353.4 (0.2) 
Epoch 108/125
train loss: 0.1029 - metric: 0.056 - time: 356.7 (3.3) 
valid loss: 0.2459 - metric: 0.061 - time: 356.7 (0.2) 
Epoch 109/125
train loss: 0.1029 - metric: 0.056 - time: 360.1 (3.4) 
valid loss: 0.2383 - metric: 0.060 - time: 360.1 (0.2) 
Epoch 110/125
train loss: 0.1024 - metric: 0.056 - time: 363.4 (3.3) 
valid loss: 0.2434 - metric: 0.060 - time: 363.4 (0.2) 
Epoch 111/125
train loss: 0.1021 - metric: 0.055 - time: 366.7 (3.3) 
valid loss: 0.2407 - metric: 0.059 - time: 366.7 (0.2) 
Epoch 112/125
train loss: 0.1024 - metric: 0.056 - time: 370.0 (3.3) 
valid loss: 0.2479 - metric: 0.061 - time: 370.0 (0.2) 
Epoch 113/125
train loss: 0.1023 - metric: 0.056 - time: 373.3 (3.3) 
valid loss: 0.2397 - metric: 0.060 - time: 373.3 (0.2) 
Epoch 114/125
train loss: 0.1021 - metric: 0.055 - time: 376.6 (3.3) 
valid loss: 0.2438 - metric: 0.061 - time: 376.6 (0.2) 
Epoch 115/125
train loss: 0.1028 - metric: 0.056 - time: 379.8 (3.3) 
valid loss: 0.2457 - metric: 0.060 - time: 379.8 (0.2) 
Epoch 116/125
train loss: 0.1021 - metric: 0.055 - time: 383.1 (3.3) 
valid loss: 0.2453 - metric: 0.060 - time: 383.1 (0.2) 
Epoch 117/125
train loss: 0.1016 - metric: 0.055 - time: 386.5 (3.4) 
valid loss: 0.2412 - metric: 0.060 - time: 386.5 (0.2) 
Epoch 118/125
train loss: 0.1021 - metric: 0.055 - time: 389.8 (3.3) 
valid loss: 0.2466 - metric: 0.061 - time: 389.8 (0.2) 
Epoch 119/125
train loss: 0.1019 - metric: 0.055 - time: 393.1 (3.3) 
valid loss: 0.2431 - metric: 0.059 - time: 393.1 (0.2) 
Epoch 120/125
train loss: 0.1016 - metric: 0.055 - time: 396.5 (3.3) 
valid loss: 0.2429 - metric: 0.060 - time: 396.5 (0.2) 
Epoch 121/125
train loss: 0.1017 - metric: 0.055 - time: 399.8 (3.3) 
valid loss: 0.2431 - metric: 0.060 - time: 399.8 (0.2) 
Epoch 122/125
train loss: 0.1016 - metric: 0.055 - time: 403.1 (3.3) 
valid loss: 0.2417 - metric: 0.060 - time: 403.1 (0.2) 
Epoch 123/125
train loss: 0.1017 - metric: 0.055 - time: 406.4 (3.4) 
valid loss: 0.2428 - metric: 0.060 - time: 406.4 (0.2) 
Epoch 124/125
train loss: 0.1020 - metric: 0.055 - time: 409.7 (3.3) 
valid loss: 0.2407 - metric: 0.060 - time: 409.7 (0.2) 
Epoch 125/125
train loss: 0.1013 - metric: 0.055 - time: 413.0 (3.3) 
valid loss: 0.2412 - metric: 0.060 - time: 413.0 (0.2) 

best epoch: 98 - metric: 0.059030607859060905 - loss: 0.23694891297877047

"""