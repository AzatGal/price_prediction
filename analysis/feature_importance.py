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
    path = '/Users/azatgalautdinov/Desktop/price_prediction/v2'
    with open(os.path.join(path, 'logs', 'config.json'), 'r') as f:
        model_cfg = json.load(f)['model_cfg']

    cfg.model_cfg = model_cfg
    trainer = Trainer(cfg, False)
    trainer.load_model(os.path.join(path, 'PricePrediction.pt'))
    model = trainer.model
    # print(trainer.model.embed.weight.dtype)

    def kv_compressor_weights(compressor):
        if isinstance(compressor, nn.ModuleList):
            return sum([x.weight.abs().sum(dim=0).cpu().numpy() for x in compressor])
        elif isinstance(compressor, nn.Linear):
            return compressor.weight.abs().sum(dim=0).cpu().numpy()
        raise NotImplementedError

    if isinstance(model.kv_compressors, nn.ModuleList):
        w = np.sum(
            [
                kv_compressor_weights(compressor)
                for i, compressor in enumerate(model.kv_compressors)
            ],
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
    print(np.array(features[0 if cfg.data_cfg.include_target else 1:])[ids])


if __name__ == '__main__':
    main()

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
