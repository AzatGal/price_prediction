import os

import pandas as pd
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
    # путь до файла с конфигом и весами модели
    path = '/Users/azatgalautdinov/Desktop/price_prediction'
    with open(os.path.join(path, 'logs', 'config.json'), 'r') as f:
        # только для модели, в data_cfg есть python class
        model_cfg = json.load(f)['model_cfg']
    cfg.model_cfg = model_cfg
    trainer = Trainer(cfg, False)
    trainer.load_model(os.path.join(path, 'PricePrediction.pt'))
    model = trainer.model
    dt = trainer.data_transformer

    i = 9
    df = pd.read_csv('/Users/azatgalautdinov/PycharmProjects/price_prediction/data/datasets/test.csv')

    # print(df[['Стоимость', 'Этаж', 'Этажей в доме', 'Площадь кухни']].corr())

    print(df.iloc[i])
    x = (torch.as_tensor(dt.transform(df)[i, 1:])
         .unsqueeze(0)
         .to(device=model.embed.weight.device))
    # print(x)
    print('Предсказанная цена: ', int(
        dt.inverse_transform(
            model(x).cpu(),
            target='num'
        )[0][0]
    ))


if __name__ == '__main__':
    main()
