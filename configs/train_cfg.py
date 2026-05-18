import os
import math
from datetime import datetime

from easydict import EasyDict
# from configs.data_cfg import cfg as data_cfg
from configs.model_cfg import cfg as model_cfg


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()

cfg.seed = 0
cfg.exp_dir = os.path.join(ROOT_DIR, 'runs',
                           datetime.now().strftime("%d-%m_%H-%M"))
cfg.batch_size = 32 # 128  # 32
cfg.num_epoch = 100

cfg.wu_ratio = 0.05
cfg.decay_ratio = 0.85
cfg.lr = 6e-4 * math.sqrt(cfg.batch_size / 256)
cfg.lr_decay_factor = 0.05
cfg.lr_decay = 'cosine'  # cosine linear

cfg.optim = 'AdamW'
cfg.optim_args = {}
cfg.weight_decay = 3e-4  # 0.1

cfg.loss = 'MSELoss'  # SmoothL1Loss L1Loss MSELoss LogCosh HuberLoss CrossEntropyLoss
cfg.loss_args = {}

cfg.accelerator_args = {}

cfg.model_cfg = model_cfg
# cfg.data_cfg = data_cfg

