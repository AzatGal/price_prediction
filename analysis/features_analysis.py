import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import json

from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

from executors.trainer import Trainer
from configs.train_cfg import cfg


@torch.no_grad()
def main():
    # path = '/Users/azatgalautdinov/Desktop/price_prediction'
    # with open(os.path.join(path, 'logs', 'config.json'), 'r') as f:
    #     model_cfg = json.load(f)['model_cfg']
    #
    # cfg.model_cfg = model_cfg
    # trainer = Trainer(cfg, False)
    # trainer.load_model(os.path.join(path, 'PricePrediction.pt'))
    # embed = trainer.model.embed
    #
    # def kv_compressor_weights(compressor):
    #     if isinstance(compressor, nn.ModuleList):
    #         return sum([x.weight.abs().sum(dim=0).cpu().numpy() for x in compressor])
    #     elif isinstance(compressor, nn.Linear):
    #         return compressor.weight.abs().sum(dim=0).cpu().numpy()
    #     raise NotImplementedError
    #
    #
    #
    # ids = np.argsort(w)[::-1]
    # features = (cfg.data_cfg.data_transformer.num_cols +
    #             cfg.data_cfg.data_transformer.cat_cols)
    # print(w[ids])
    # print(np.array(features[0 if cfg.data_cfg.include_target else 1:])[ids])

    cfg.load_checkpoint = '/Users/azatgalautdinov/Desktop/price_prediction/best/checkpoint'
    trainer = Trainer(cfg)
    print(trainer.optimizer.param_groups[0]['lr'])



if __name__ == '__main__':
    main()
