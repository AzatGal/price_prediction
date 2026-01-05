import os

import dill
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from easydict import EasyDict
from torch.utils.data import Dataset


class ApartmentDataset(Dataset):
    def __init__(self,
                 dataset_type,
                 path,
                 data_transformer,
                 target,
                 smooth=False
                 ) -> None:
        df = pd.read_csv(os.path.join(path, f"{dataset_type}.csv"))
        self.features = torch.as_tensor(data_transformer.transform(df))
        self.label = torch.as_tensor(df[['Стоимость']].values)

        if target == 'cat':
            l = self.features[:, 0]
            c = data_transformer.num_bins[0]
            if smooth:
                s = 15  # 7
                v = torch.softmax(torch.signal.windows.gaussian(s) * 7, dim=0)
                r = len(self.features)
                t = torch.zeros(r, c)
                for i in range(r):
                    li = l[i].item()
                    lb = max(li - s // 2, 0)
                    rb = min(li + s // 2 + 1, c)
                    t[i, lb: rb] = v[max(0, s // 2 - li + lb): min(s, s // 2 - li + rb)]
                self.target = t
            else:
                self.target = l  # .float()
        elif target == 'num':
            self.target = torch.as_tensor(data_transformer.transform(df[['Стоимость']], target=True))
        else:
            raise NotImplementedError()

        self.mask = torch.zeros(self.features.shape[1], dtype=torch.bool)
        self.mask[0] = True

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {'features': self.features[idx],
                'mask': self.mask,
                'target': self.target[idx],
                # torch.as_tensor(self.target[idx]).float(),
                'label': self.label[idx]}


if __name__ == '__main__':
    from configs.data_cfg import cfg
    ad = ApartmentDataset('train', **cfg)
    t = ad[0]
    print(cfg.data_transformer.inverse_transform(t['target'], target=True))
    print(t['label'])
