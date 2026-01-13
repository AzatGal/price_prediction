import os

from easydict import EasyDict
from configs.data_cfg import cfg as data_cfg
from configs.model_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()
cfg.seed = 0
cfg.exp_dir = os.path.join(ROOT_DIR, 'exp_dir')

cfg.batch_size = 8 * 1024  # 16
cfg.num_epoch = 100

cfg.lr = 1e-2  # 1 3 5
cfg.lr_decay_factor = 0.01

cfg.weight_decay = 1e-5
cfg.lr_decay = 'cosine'  # cosine

model_cfg.pred_dim = 1  # model_cfg.num_embed_features[0]  # 1  #
model_cfg.log_softmax = False  # False True
cfg.loss = 'MSELoss'  # CrossEntropyLoss SmoothL1Loss KLDivLoss L1Loss MSELoss
cfg.loss_args = {}  # 'reduction': 'batchmean'}

cfg.accelerator_args = {'mixed_precision': 'fp16', 'cpu': True}

cfg.target_type = 'num'  # cat num

# cfg.lr_decay_by_block = 0.95
cfg.load_pretrained = os.path.join(ROOT_DIR, 'exp_dir', 'MaskedTableAutoencoder.pt')
# MaskedTableAutoencoder MaskedTableModeling

cfg.model = 'PricePrediction'  # PricePrediction TabM PricePredEnsemble

data_cfg.target_type = cfg.target_type
data_cfg.mask_first_token = len(data_cfg.features) == len(model_cfg.num_embed_features)

cfg.model_cfg = model_cfg
cfg.data_cfg = data_cfg


"""
cpu
Epoch 1/300
train loss: 1.1867 - metric: 0.468 - time: 7.1 (7.1) 
valid loss: 1.0305 - metric: 0.584 - time: 7.1 (0.3) 
best
Epoch 2/300
train loss: 0.8239 - metric: 0.446 - time: 13.9 (6.8) 
valid loss: 0.5455 - metric: 0.293 - time: 13.9 (0.3) 
best
Epoch 3/300
train loss: 0.4561 - metric: 0.297 - time: 22.9 (9.0) 
valid loss: 0.2994 - metric: 0.213 - time: 22.9 (0.3) 
best
Epoch 4/300
train loss: 0.2505 - metric: 0.204 - time: 32.0 (9.2) 
valid loss: 0.1633 - metric: 0.173 - time: 32.0 (0.3) 
best
Epoch 5/300
train loss: 0.1638 - metric: 0.166 - time: 43.3 (11.3) 
valid loss: 0.1062 - metric: 0.135 - time: 43.3 (0.3) 
best
Epoch 6/300
train loss: 0.1179 - metric: 0.140 - time: 51.5 (8.1) 
valid loss: 0.0879 - metric: 0.112 - time: 51.5 (0.3) 
best
Epoch 7/300
train loss: 0.0869 - metric: 0.117 - time: 59.4 (7.9) 
valid loss: 0.0645 - metric: 0.097 - time: 59.4 (0.3) 
best
Epoch 8/300
train loss: 0.0708 - metric: 0.105 - time: 67.1 (7.7) 
valid loss: 0.0569 - metric: 0.100 - time: 67.1 (0.4) 
Epoch 9/300
train loss: 0.0614 - metric: 0.098 - time: 76.3 (9.2) 
valid loss: 0.0481 - metric: 0.088 - time: 76.3 (0.4) 
best
Epoch 10/300
train loss: 0.0557 - metric: 0.092 - time: 84.1 (7.8) 
valid loss: 0.0492 - metric: 0.092 - time: 84.1 (0.3) 
Epoch 11/300
train loss: 0.0525 - metric: 0.091 - time: 91.9 (7.8) 
valid loss: 0.0504 - metric: 0.085 - time: 91.9 (0.4) 
best
Epoch 12/300
train loss: 0.0502 - metric: 0.087 - time: 99.5 (7.6) 
valid loss: 0.0473 - metric: 0.090 - time: 99.5 (0.3) 
Epoch 13/300
train loss: 0.0474 - metric: 0.085 - time: 107.3 (7.7) 
valid loss: 0.0412 - metric: 0.077 - time: 107.3 (0.4) 
best
Epoch 14/300
train loss: 0.0466 - metric: 0.084 - time: 115.6 (8.3) 
valid loss: 0.0423 - metric: 0.084 - time: 115.6 (0.3) 
Epoch 15/300
train loss: 0.0442 - metric: 0.082 - time: 123.4 (7.8) 
valid loss: 0.0437 - metric: 0.088 - time: 123.4 (0.3) 
Epoch 16/300
train loss: 0.0456 - metric: 0.084 - time: 131.1 (7.7) 
valid loss: 0.0394 - metric: 0.077 - time: 131.1 (0.3) 
Epoch 17/300
train loss: 0.0430 - metric: 0.081 - time: 139.0 (7.9) 
valid loss: 0.0395 - metric: 0.075 - time: 139.0 (0.3) 
best
Epoch 18/300
train loss: 0.0412 - metric: 0.079 - time: 148.3 (9.3) 
valid loss: 0.0371 - metric: 0.072 - time: 148.3 (0.3) 
best
Epoch 19/300
train loss: 0.0393 - metric: 0.077 - time: 155.8 (7.4) 
valid loss: 0.0367 - metric: 0.073 - time: 155.8 (0.3) 
Epoch 20/300
train loss: 0.0404 - metric: 0.079 - time: 163.5 (7.8) 
valid loss: 0.0367 - metric: 0.072 - time: 163.5 (0.3) 
best
Epoch 21/300
train loss: 0.0379 - metric: 0.076 - time: 171.6 (8.1) 
valid loss: 0.0361 - metric: 0.075 - time: 171.6 (0.3) 
Epoch 22/300
train loss: 0.0380 - metric: 0.077 - time: 180.1 (8.4) 
valid loss: 0.0357 - metric: 0.072 - time: 180.1 (0.3) 
best
Epoch 23/300
train loss: 0.0376 - metric: 0.076 - time: 187.7 (7.6) 
valid loss: 0.0363 - metric: 0.076 - time: 187.7 (0.3) 
Epoch 24/300
train loss: 0.0367 - metric: 0.075 - time: 195.2 (7.5) 
valid loss: 0.0371 - metric: 0.072 - time: 195.2 (0.4) 
Epoch 25/300
train loss: 0.0360 - metric: 0.074 - time: 202.6 (7.5) 
valid loss: 0.0342 - metric: 0.070 - time: 202.6 (0.4) 
best
Epoch 26/300
train loss: 0.0340 - metric: 0.072 - time: 210.2 (7.5) 
valid loss: 0.0332 - metric: 0.070 - time: 210.2 (0.3) 
best
Epoch 27/300
train loss: 0.0332 - metric: 0.071 - time: 218.3 (8.1) 
valid loss: 0.0346 - metric: 0.069 - time: 218.3 (0.3) 
best
Epoch 28/300
train loss: 0.0327 - metric: 0.070 - time: 225.9 (7.6) 
valid loss: 0.0335 - metric: 0.072 - time: 225.9 (0.3) 
Epoch 29/300
train loss: 0.0333 - metric: 0.072 - time: 234.7 (8.8) 
valid loss: 0.0315 - metric: 0.067 - time: 234.7 (0.3) 
best
Epoch 30/300
train loss: 0.0313 - metric: 0.069 - time: 242.4 (7.8) 
valid loss: 0.0314 - metric: 0.066 - time: 242.4 (0.4) 
best
Epoch 31/300
train loss: 0.0307 - metric: 0.068 - time: 250.2 (7.7) 
valid loss: 0.0316 - metric: 0.067 - time: 250.2 (0.3) 
Epoch 32/300
train loss: 0.0307 - metric: 0.068 - time: 257.8 (7.7) 
valid loss: 0.0317 - metric: 0.067 - time: 257.8 (0.3) 
Epoch 33/300
train loss: 0.0305 - metric: 0.068 - time: 265.5 (7.7) 
valid loss: 0.0364 - metric: 0.071 - time: 265.5 (0.3) 
Epoch 34/300
train loss: 0.0354 - metric: 0.075 - time: 273.5 (8.0) 
valid loss: 0.0369 - metric: 0.078 - time: 273.5 (0.3) 
Epoch 35/300
train loss: 0.0345 - metric: 0.074 - time: 282.2 (8.7) 
valid loss: 0.0336 - metric: 0.070 - time: 282.2 (0.4) 
Epoch 36/300
train loss: 0.0323 - metric: 0.071 - time: 291.4 (9.1) 
valid loss: 0.0318 - metric: 0.067 - time: 291.4 (0.4) 
Epoch 37/300
train loss: 0.0311 - metric: 0.069 - time: 299.8 (8.4) 
valid loss: 0.0328 - metric: 0.070 - time: 299.8 (0.5) 
Epoch 38/300
train loss: 0.0301 - metric: 0.068 - time: 307.5 (7.7) 
valid loss: 0.0312 - metric: 0.065 - time: 307.5 (0.3) 
best
Epoch 39/300
train loss: 0.0300 - metric: 0.068 - time: 315.6 (8.1) 
valid loss: 0.0310 - metric: 0.068 - time: 315.6 (0.3) 
Epoch 40/300
train loss: 0.0294 - metric: 0.068 - time: 323.6 (8.0) 
valid loss: 0.0302 - metric: 0.065 - time: 323.6 (0.3) 
best
Epoch 41/300
train loss: 0.0281 - metric: 0.065 - time: 332.1 (8.5) 
valid loss: 0.0303 - metric: 0.065 - time: 332.1 (0.3) 
best
Epoch 42/300
train loss: 0.0278 - metric: 0.065 - time: 340.0 (7.9) 
valid loss: 0.0298 - metric: 0.066 - time: 340.0 (0.3) 
Epoch 43/300
train loss: 0.0276 - metric: 0.066 - time: 348.9 (8.9) 
valid loss: 0.0298 - metric: 0.064 - time: 348.9 (0.4) 
best
Epoch 44/300
train loss: 0.0271 - metric: 0.065 - time: 357.1 (8.2) 
valid loss: 0.0286 - metric: 0.063 - time: 357.1 (0.2) 
best
Epoch 45/300
train loss: 0.0265 - metric: 0.064 - time: 365.7 (8.6) 
valid loss: 0.0297 - metric: 0.064 - time: 365.7 (0.3) 
Epoch 46/300
train loss: 0.0268 - metric: 0.064 - time: 373.9 (8.2) 
valid loss: 0.0290 - metric: 0.064 - time: 373.9 (0.3) 
Epoch 47/300
train loss: 0.0267 - metric: 0.065 - time: 382.0 (8.1) 
valid loss: 0.0299 - metric: 0.064 - time: 382.0 (0.3) 
Epoch 48/300
train loss: 0.0262 - metric: 0.064 - time: 390.2 (8.2) 
valid loss: 0.0289 - metric: 0.064 - time: 390.2 (0.3) 
Epoch 49/300
train loss: 0.0262 - metric: 0.064 - time: 398.4 (8.2) 
valid loss: 0.0290 - metric: 0.066 - time: 398.4 (0.4) 
Epoch 50/300
train loss: 0.0267 - metric: 0.065 - time: 407.5 (9.2) 
valid loss: 0.0312 - metric: 0.065 - time: 407.5 (0.3) 
Epoch 51/300
train loss: 0.0263 - metric: 0.064 - time: 415.2 (7.6) 
valid loss: 0.0288 - metric: 0.064 - time: 415.2 (0.3) 
Epoch 52/300
train loss: 0.0255 - metric: 0.063 - time: 422.8 (7.7) 
valid loss: 0.0291 - metric: 0.065 - time: 422.8 (0.3) 
Epoch 53/300
train loss: 0.0253 - metric: 0.063 - time: 430.8 (8.0) 
valid loss: 0.0295 - metric: 0.065 - time: 430.8 (0.3) 
Epoch 54/300
train loss: 0.0251 - metric: 0.063 - time: 439.1 (8.3) 
valid loss: 0.0294 - metric: 0.064 - time: 439.1 (0.3) 
Epoch 55/300
train loss: 0.0249 - metric: 0.062 - time: 447.2 (8.2) 
valid loss: 0.0299 - metric: 0.064 - time: 447.2 (0.3) 
Epoch 56/300
train loss: 0.0246 - metric: 0.062 - time: 455.5 (8.2) 
valid loss: 0.0287 - metric: 0.063 - time: 455.5 (0.3) 
Epoch 57/300
train loss: 0.0246 - metric: 0.062 - time: 464.9 (9.4) 
valid loss: 0.0280 - metric: 0.061 - time: 464.9 (0.4) 
best
Epoch 58/300
train loss: 0.0239 - metric: 0.062 - time: 472.9 (8.1) 
valid loss: 0.0283 - metric: 0.061 - time: 472.9 (0.3) 
best
Epoch 59/300
train loss: 0.0241 - metric: 0.062 - time: 481.5 (8.5) 
valid loss: 0.0281 - metric: 0.061 - time: 481.5 (0.3) 
Epoch 60/300
train loss: 0.0234 - metric: 0.061 - time: 490.3 (8.8) 
valid loss: 0.0279 - metric: 0.061 - time: 490.3 (0.3) 
best
Epoch 61/300
train loss: 0.0232 - metric: 0.061 - time: 498.0 (7.7) 
valid loss: 0.0281 - metric: 0.061 - time: 498.0 (0.3) 
Epoch 62/300
train loss: 0.0233 - metric: 0.061 - time: 506.1 (8.1) 
valid loss: 0.0286 - metric: 0.062 - time: 506.1 (0.3) 
Epoch 63/300
train loss: 0.0235 - metric: 0.061 - time: 514.0 (7.9) 
valid loss: 0.0280 - metric: 0.062 - time: 514.0 (0.3) 
Epoch 64/300
train loss: 0.0236 - metric: 0.061 - time: 523.3 (9.3) 
valid loss: 0.0283 - metric: 0.062 - time: 523.3 (0.3) 
Epoch 65/300
train loss: 0.0239 - metric: 0.062 - time: 530.9 (7.6) 
valid loss: 0.0289 - metric: 0.063 - time: 530.9 (0.3) 
Epoch 66/300
train loss: 0.0243 - metric: 0.062 - time: 538.9 (8.0) 
valid loss: 0.0298 - metric: 0.065 - time: 538.9 (0.3) 
Epoch 67/300
train loss: 0.0247 - metric: 0.063 - time: 546.8 (7.9) 
valid loss: 0.0289 - metric: 0.063 - time: 546.8 (0.3) 
Epoch 68/300
train loss: 0.0236 - metric: 0.062 - time: 554.7 (8.0) 
valid loss: 0.0287 - metric: 0.063 - time: 554.7 (0.3) 
Epoch 69/300
train loss: 0.0237 - metric: 0.062 - time: 564.5 (9.7) 
valid loss: 0.0281 - metric: 0.062 - time: 564.5 (0.3) 
Epoch 70/300

"""