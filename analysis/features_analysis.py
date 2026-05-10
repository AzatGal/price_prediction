import os
import copy

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
    path = '/datasets/custom_dataset/datasets/'
    train = pd.read_csv(os.path.join(path, 'train.csv')).drop(columns=['Unnamed: 0'])
    print(train.info())
    # valid = pd.read_csv(os.path.join(path, 'val.csv')).drop(columns=['Unnamed: 0'])
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
    path = '/Users/azatgalautdinov/PycharmProjects/price_prediction/runs/train/16-02_23-50'
    with open(os.path.join(path, 'logs', 'config.json'), 'r') as f:
        model_cfg = json.load(f)['model_cfg']

    cfg.model_cfg = model_cfg
    trainer = Trainer(cfg, False)
    # trainer.load_model('/Users/azatgalautdinov/PycharmProjects/price_prediction/executors/test.pt')
    # w1 = copy.deepcopy(trainer.model.blocks[0].attn.qkv_proj.weight)
    w1 = copy.deepcopy(trainer.model.kv_compressors[0][1].weight)

    trainer.load_model(os.path.join(path, 'TablePredictor.pt'))
    # w2 = copy.deepcopy(trainer.model.blocks[0].attn.qkv_proj.weight)
    w2 = copy.deepcopy(trainer.model.kv_compressors[0][1].weight)
    model = trainer.model

    print(torch.abs(w2 - w1).mean())
    # print(w1)
    # print(w2)
    model(torch.ones(1, 19, dtype=torch.long, device=next(model.parameters()).device))

    # print('qkv', torch.all(model.blocks[0].attn.qkv_proj.weight).item())
    # print('k', torch.all(model.kv_compressors[0][0].weight).item())
    # print('v', torch.all(model.kv_compressors[0][1].weight).item())
    # print('o', torch.all(model.blocks[0].attn.out_proj.weight == 0).item())


    # cfg.load_checkpoint = '/Users/azatgalautdinov/Desktop/price_prediction/best/checkpoint'
    # trainer = Trainer(cfg)
    # print(trainer.optimizer.param_groups[0]['lr'])


if __name__ == '__main__':
    main()
    # data_analysis()
    # feature_analysis()
