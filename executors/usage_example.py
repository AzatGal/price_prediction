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
    path = '/Users/azatgalautdinov/Desktop/price_prediction/transformer'
    with open(os.path.join(path, 'logs', 'config.json'), 'r') as f:
        # только для модели, в data_cfg есть python class
        cfg.model_cfg = json.load(f)['model_cfg']

    cfg.model = 'TablePredictor'
    trainer = Trainer(cfg, False)
    trainer.load_model(os.path.join(path, 'TablePredictor.pt'))

    # device = torch.device('cpu')

    model = trainer.model  # .to(device=device)
    # model.load_state_dict(
    #     torch.load(os.path.join('/Users/azatgalautdinov/Desktop/price_prediction/best',
    #                             'TablePredictor.pt')
    #     , map_location=device)
    # )

    print("dtype: ", next(model.parameters()).dtype,
          "\ndevice: ", next(model.parameters()).device)

    dt = trainer.data_transformer

    i = 9
    df = pd.read_csv('/dataset/apartment_dataset/datasets/test.csv')

    print('\n', df.iloc[i])

    t = dt.transform(df)[i, 1:]
    print(t)
    x = (torch.as_tensor(t)
         .unsqueeze(0)
         # .to(device=device))
         .to(device=next(model.parameters()).device))
    # print(x)
    print('\nПредсказанная цена: ', int(
        dt.inverse_transform(
            model(x).cpu(),
            target='num'
        )[0][0]
    ))


if __name__ == '__main__':
    main()
