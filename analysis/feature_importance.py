import os
import math
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
def get_feature_importance(model):
    # path = '/Users/azatgalautdinov/Desktop/price_prediction/v2'
    # with open(os.path.join(path, 'logs', 'config.json'), 'r') as f:
    #     model_cfg = json.load(f)['model_cfg']

    # cfg.model_cfg = model_cfg

    # print(trainer.model.embed.weight.dtype)

    def kv_compressor_weights(compressor):
        if isinstance(compressor, nn.ModuleList):
            return sum([x.weight.abs().mean(dim=0).cpu().numpy() for x in compressor])
            # return sum([x.weight.norm(dim=0).cpu().numpy() for x in compressor])
        elif isinstance(compressor, nn.Linear):
            return compressor.weight.abs().mean(dim=0).cpu().numpy()
            # return compressor.weight.norm(dim=0).cpu().numpy()
        raise NotImplementedError

    if isinstance(model.kv_compressors, nn.ModuleList):
        w = np.sum(
            [kv_compressor_weights(compressor)
             for i, compressor in enumerate(model.kv_compressors)],
            axis=0
        )
    elif isinstance(model.kv_compressors, nn.Linear):
        w = kv_compressor_weights(model.kv_compressors)
    else:
        raise NotImplementedError

    ids = np.argsort(w)[::-1]
    features = (cfg.data_cfg.data_transformer.num_cols +
                cfg.data_cfg.data_transformer.cat_cols)
    print(w[ids])
    print(np.array(features[int(not cfg.data_cfg.include_target):])[ids])


@torch.no_grad()
def get_feature_importance_v2(model):
    # path = '/Users/azatgalautdinov/Desktop/price_prediction/v2'
    # with open(os.path.join(path, 'logs', 'config.json'), 'r') as f:
    #     model_cfg = json.load(f)['model_cfg']

    # cfg.model_cfg = model_cfg
    # print(trainer.model.embed.weight.dtype)

    def importance(compressor, t):
        # if isinstance(compressor, nn.ModuleList):
        #     w = (compressor[0].weight.abs() +
        #          compressor[1].weight.abs())
        # elif isinstance(compressor, nn.Linear):
        #     w = compressor.weight.abs()
        # else:
        #     raise Exception

        w = compressor.weight.abs()
        # print(t, torch.all(w == 0).item())
        # mean_amplitude = w.mean(dim=0)
        # print('mean_amplitude', torch.all(mean_amplitude == 0).item())
        entropy = -torch.sum(w * torch.log(w + 1e-8), dim=0)
        normalized_entropy = entropy / math.log(w.shape[0] + 1e-8)
        # print('normalized_entropy', torch.isnan(normalized_entropy).any().item())

        importance = w.mean(dim=0) * (1 + normalized_entropy)
        importance = importance / importance.mean()
        # print('importance', torch.isnan(importance).any().item())
        return importance.cpu().numpy()

    w = np.sum(
        [
            (importance(kv_compressors[0], 'k') +
             importance(kv_compressors[1], 'v')) * (0.95 ** i)
            for i, kv_compressors in enumerate(model.kv_compressors)
        ],
        axis=0
    )

    ids = np.argsort(w)[::-1]
    features = (cfg.data_cfg.data_transformer.num_cols +
                cfg.data_cfg.data_transformer.cat_cols)
    print(w[ids])
    print(np.array(features[int(not cfg.data_cfg.include_target):])[ids])


@torch.no_grad()
def main():
    path = '/Users/azatgalautdinov/Desktop/price_prediction/best'
    # with open(os.path.join(path, 'logs', 'config.json'), 'r') as f:
    #     model_cfg = json.load(f)['model_cfg']
    trainer = Trainer(cfg, False)
    trainer.load_model(os.path.join(path, 'TablePredictor.pt'))
    model = trainer.model

    def kv_compressor_weights(compressor):
        if isinstance(compressor, nn.ModuleList):
            return sum([x.weight.abs().sum(dim=0).cpu().numpy() for x in compressor])
        elif isinstance(compressor, nn.Linear):
            return compressor.weight.abs().sum(dim=0).cpu().numpy()
        raise NotImplementedError

    # if isinstance(model.kv_compressors, nn.ModuleList):
    #     w = np.sum(
    #         [
    #             kv_compressor_weights(compressor) / ((i + 1) ** 2)
    #             for i, compressor in enumerate(model.kv_compressors)
    #         ],
    #         axis=0
    #     )

    w1 = kv_compressor_weights(model.kv_compressors[0])
    for i in range(1, len(model.kv_compressors)):
        w2 = kv_compressor_weights(model.kv_compressors[i])
        print(w2 - w1)
        w1 = w2


if __name__ == '__main__':
    trainer = Trainer(cfg, False)
    trainer.load_model(os.path.join(
        # '/Users/azatgalautdinov/Desktop/price_prediction/best',
        '/Users/azatgalautdinov/PycharmProjects/price_prediction/runs/train/14-02_14-46',
        'TablePredictor.pt'
    ))
    model = trainer.model
    # get_feature_importance()
    get_feature_importance_v2()


"""
[2.7843444  2.777239   2.2842464  2.1284864  2.0257008  1.9084042
 1.7282344  1.4424212  1.2407185  1.2059945  1.1965979  1.1403681
 1.068266   1.0404059  0.997353   0.96044624 0.9294794  0.9158576
 0.48160803]
['Общая площадь' 'Этаж' 'Район' 'Округ' 'Площадь кухни' 'Этажей в доме'
 'Жилая площадь' 'Расстояние до метро' 'Тип продажи' 'Вид из окон'
 'Высота потолков' 'Количество комнат' 'Кол-во раздельных санузлов'
 'Тип дома' 'Объект продажи' 'Лифт пассажирский (кол-во)' 'Парковка'
 'Лифт грузовой (кол-во)' 'Мусоропровод']
"""
