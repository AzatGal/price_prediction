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
cfg.batch_size = 2 * 1024
cfg.num_epoch = 400

cfg.wu_ratio = 0.05
cfg.decay_ratio = 0.75
cfg.lr = 6e-4 * cfg.batch_size / 256  # 1e-3 bs: 8 * 1024
cfg.lr_decay_factor = 1e-2
cfg.lr_decay = 'cosine'  # cosine linear

cfg.optim = 'AdamW'
cfg.optim_args = {}
cfg.weight_decay = 1e-5

cfg.loss = 'MSELoss'  # SmoothL1Loss  L1Loss MSELoss
cfg.loss_args = {}  # 'reduction': 'batchmean'}

cfg.accelerator_args = {}  # 'mixed_precision': 'fp16'}  # , 'cpu': True}

# cfg.lr_decay_by_block = 0.95
# cfg.load_pretrained = os.path.join(ROOT_DIR, 'runs', 'MaskedTableAutoencoder.pt')
# MaskedTableAutoencoder MaskedTableModeling

# cfg.load_checkpoint = os.path.join(ROOT_DIR, 'runs', 'train', '16-01_16-04', "checkpoint")

cfg.model = 'PricePrediction'

cfg.task = 'train'
model_cfg.pred_dim = 1
data_cfg.include_target = model_cfg.mask_first_token
data_cfg.task = cfg.task

cfg.model_cfg = model_cfg
cfg.data_cfg = data_cfg

