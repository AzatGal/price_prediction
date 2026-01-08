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

cfg.loss = 'CrossEntropyLoss'  # CrossEntropyLoss SmoothL1Loss KLDivLoss L1Loss
cfg.loss_args = {}  # 'reduction': 'batchmean'}
cfg.decay = 'linear'

cfg.accelerator_args = {'mixed_precision': 'fp16', 'cpu': True}

cfg.data_cfg = data_cfg

cfg.exp_dir = os.path.join(ROOT_DIR, 'exp_dir', 'pretrain')
cfg.num_epoch = 125

cfg.model = 'MaskedTableAutoencoder'
cfg.num_masks = int(0.75 * len(data_cfg.features))
model_cfg.decoder_embed_dim = model_cfg.embed_dim // 2
model_cfg.decoder_num_heads = model_cfg.num_heads
model_cfg.decoder_num_blocks = min(1, model_cfg.num_blocks // 3)

model_cfg.pred_dim = sum(model_cfg.num_embed_features)  # 1  #
cfg.model_cfg = model_cfg

# cfg.model = 'MaskedTableModeling'
# cfg.num_masks = int(0.4 * len(data_cfg.features))

