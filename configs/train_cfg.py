import os
from datetime import datetime

from easydict import EasyDict
from configs.data_cfg import cfg as data_cfg
from configs.model_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()

cfg.seed = 0
cfg.exp_dir = os.path.join(ROOT_DIR, 'runs', 'train',
                           datetime.now().strftime("%d-%m_%H-%M"))
cfg.batch_size = 1024
cfg.num_epoch = 400

cfg.wu_ratio = 0.075
cfg.decay_ratio = 0.75
cfg.lr = 6e-4 * cfg.batch_size / 256  # 1e-3 bs: 8 * 1024
cfg.lr_decay_factor = 1e-2
cfg.lr_decay = 'linear'  # cosine linear

cfg.optim = 'AdamW'
cfg.optim_args = {}
cfg.weight_decay = 3e-4

cfg.loss = 'HuberLoss'  # SmoothL1Loss L1Loss MSELoss LogCosh HuberLoss
cfg.loss_args = {'delta': 8}  # 'reduction': 'batchmean'}

cfg.accelerator_args = {'mixed_precision': 'fp16'}  # , 'cpu': True}

# cfg.lr_decay_by_block = 0.95
# cfg.load_pretrained = os.path.join(ROOT_DIR, 'runs', 'MaskedTableAutoencoder.pt')
# MaskedTableAutoencoder MaskedTableModeling

# cfg.load_checkpoint = os.path.join(ROOT_DIR, 'runs', 'train', '18-01_23-42', "checkpoint")

cfg.model = 'PricePrediction'

cfg.task = 'train'
model_cfg.pred_dim = 1
data_cfg.include_target = model_cfg.mask_first_token
data_cfg.task = cfg.task

cfg.model_cfg = model_cfg
cfg.data_cfg = data_cfg

