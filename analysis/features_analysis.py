import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

import json

from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

from executors.trainer import Trainer
from configs.train_cfg import cfg


def data_analysis():
    path = '/Users/azatgalautdinov/PycharmProjects/price_prediction/data/datasets/'
    train = pd.read_csv(os.path.join(path, 'train.csv')).drop(columns=['Unnamed: 0'])
    print(train.info())
    # valid = pd.read_csv(os.path.join(path, 'valid.csv')).drop(columns=['Unnamed: 0'])
    # test = pd.read_csv(os.path.join(path, 'test.csv')).drop(columns=['Unnamed: 0'])
    # t = len(train) + len(valid) + len(test)
    # print(t)
    # print(len(train) / t)
    # print(len(valid) / t)
    # print(len(test) / t)


def feature_analysis():
    dt = cfg.data_cfg.data_transformer

    # t = dt.cat_processor.categories_.copy()
    # for i, (col, cats) in enumerate(zip(dt.cat_cols, dt.cat_processor.categories_)):
    #     t[i][pd.isna(cats)] = 'other'
    #
    for col, cats in zip(dt.cat_cols, dt.cat_processor.categories_):
        print(col, '\tКатегориальный\t', len(cats))

    for col, edges in zip(dt.num_cols, dt.num_processor.bin_edges_):
        print(col, '\tНепрерывный\t', len(edges))


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
    # main()
    # data_analysis()
    feature_analysis()
