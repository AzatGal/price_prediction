import os

import dill
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict
from torch.utils.data import Dataset


class ApartmentDataset(Dataset):
    def __init__(self,
                 path,
                 dataset_type,
                 data_transformer,
                 target_transformer
                 ) -> None:
        df = pd.read_csv(os.path.join(path, f"{dataset_type}_data.csv"))
        self.data_transformer = data_transformer
        self.target_transformer = target_transformer
        # self.features = data_transformer.transform(df)
        # self.target = target_transformer.transform(df[['Стоимость']])
        self.label = df[['Стоимость']].values
        self.mask = torch.zeros(dtype=torch.bool)
        self.mask[0] = True

    def get_features(self, df):
        data

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': torch.as_tensor(self.features[idx]),
            'mask': self.mask,
            'target': torch.as_tensor(self.target[idx]),
            # 'label': torch.as_tensor(self.label[idx])
        }
