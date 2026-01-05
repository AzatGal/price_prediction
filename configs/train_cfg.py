import os

from easydict import EasyDict
from configs.data_cfg import cfg as data_cfg
from configs.model_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()
cfg.seed = 0

cfg.batch_size = 128
cfg.lr = 1e-5  # 1 3 5
cfg.weight_decay = 1e-5

cfg.loss = 'L1Loss'  # CrossEntropyLoss SmoothL1Loss KLDivLoss L1Loss
cfg.loss_args = {}  # 'reduction': 'batchmean'}
cfg.decay = 'linear'

cfg.accelerator_args = {'mixed_precision': 'fp16', 'cpu': True}
cfg.model_cfg = model_cfg
cfg.data_cfg = data_cfg

cfg.exp_dir = os.path.join(ROOT_DIR, 'exp_dir')
cfg.num_epoch = 512

