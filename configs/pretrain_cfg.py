import os

from easydict import EasyDict
from configs.data_cfg import cfg as data_cfg
from configs.model_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()
cfg.seed = 0

cfg.batch_size = 16  # 128
cfg.lr = 3e-3  # 1 3 5
cfg.lr_decay_factor = 1e-3

cfg.weight_decay = 1e-5
cfg.target = 'mask'
cfg.num_masks = int(0.4 * len(data_cfg.features))

cfg.loss = 'CrossEntropyLoss'  # CrossEntropyLoss SmoothL1Loss KLDivLoss L1Loss
cfg.loss_args = {}  # 'reduction': 'batchmean'}
cfg.decay = 'linear'

cfg.accelerator_args = {'mixed_precision': 'fp16', 'cpu': True}

model_cfg.pred_dim = sum(model_cfg.num_embed_features)  # 1  #
cfg.model_cfg = model_cfg
cfg.data_cfg = data_cfg

cfg.exp_dir = os.path.join(ROOT_DIR, 'exp_dir', 'pretrain')
cfg.num_epoch = 256
cfg.task = 'masked_table_modeling'

