import os

from easydict import EasyDict
from configs.data_cfg import cfg as data_cfg
from configs.model_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()
cfg.seed = 0

cfg.batch_size = 16  # 16
cfg.lr = 3e-3  # 1 3 5
cfg.lr_decay_factor = 1e-3
cfg.lr_decay_by_block = 0.9

cfg.weight_decay = 1e-5
cfg.target = 'num'
cfg.num_masks = 1

cfg.loss = 'L1Loss'  # CrossEntropyLoss SmoothL1Loss KLDivLoss L1Loss
cfg.loss_args = {}  # 'reduction': 'batchmean'}
cfg.decay = 'cosine'

cfg.accelerator_args = {'mixed_precision': 'fp16', 'cpu': True}

model_cfg.pred_dim = 1  # cfg.num_embed_features[0]  # 1  #
cfg.model_cfg = model_cfg
cfg.data_cfg = data_cfg

cfg.exp_dir = os.path.join(ROOT_DIR, 'exp_dir', 'train')
cfg.load_pretrained = os.path.join(ROOT_DIR, 'exp_dir', 'pretrain', "transformer.pt")

cfg.num_epoch = 256
cfg.task = 'price_prediction'
