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
    path = '/Users/azatgalautdinov/PycharmProjects/price_prediction/runs/train/15-02_21-59'
    with open(os.path.join(path, 'logs', 'config.json'), 'r') as f:
        model_cfg = json.load(f)['model_cfg']

    cfg.model_cfg = model_cfg
    trainer = Trainer(cfg, False)
    trainer.load_model(os.path.join(path, 'TablePredictor.pt'))
    model = trainer.model

    model(torch.zeros(1, 19, dtype=torch.long, device=next(model.parameters()).device))

    # print(model.blocks[0].attn.qkv_proj.weight)
    # print(model.kv_compressors[0][0].weight)
    # print(model.kv_compressors[0][1].weight)

    # k1 = model.kv_compressors[0][0].weight
    # v1 = model.kv_compressors[0][1].weight
    #
    # trainer.load_model('/Users/azatgalautdinov/PycharmProjects/price_prediction/executors/test.pt')
    # model = trainer.model
    #
    # k2 = model.kv_compressors[0][0].weight
    # v2 = model.kv_compressors[0][1].weight
    #
    # print(k1 - k2)
    # print(v1 - v2)


    # cfg.load_checkpoint = '/Users/azatgalautdinov/Desktop/price_prediction/best/checkpoint'
    # trainer = Trainer(cfg)
    # print(trainer.optimizer.param_groups[0]['lr'])


if __name__ == '__main__':
    main()
    # data_analysis()
    # feature_analysis()
