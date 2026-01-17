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

    i = 4
    df = pd.read_csv('/Users/azatgalautdinov/PycharmProjects/price_prediction/data/datasets/valid.csv')
    print(df.iloc[i])
    x = torch.as_tensor(dt.transform(df)[i, 1:]).unsqueeze(0).to(device=model.embed.weight.device)
    # print(x)
    print('Предсказанная цена: ', int(
        dt.inverse_transform(
            model(x).cpu(),
            target='num'
        )[0][0]
    ))

    # print(
    #     trainer.valid_data[i]['features']
    # )
    # # стоимость
    # print(
    #     trainer.valid_data[i]['label']
    # )
    # df = trainer.data_transformer.inverse_transform(trainer.valid_data[i]['features'], numpy=False)
    # # 1 колонная стоимость - фиктивная тут
    # print(df[df.columns[1:]])
    # print(
    #     trainer.data_transformer.inverse_transform(
    #         trainer.model(trainer.valid_data[i]['features'].unsqueeze(0).to(device='mps')).cpu(),
    #         target='num'
    #     )
    # )


if __name__ == '__main__':
    main()
