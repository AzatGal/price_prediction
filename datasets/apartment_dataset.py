import dill
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ApartmentDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 ct_path: str,
                 target_path: str = None,
                 num_mask_features: int = None
                 ) -> None:

        df = pd.read_csv(data_path)
        self.size = len(df)

        with open(ct_path, 'rb') as f:
            ct = dill.load(f)

        self.data = ct.transform(df)
        self.num_mask_features = num_mask_features

        if target_path is not None:
            self.mask = torch.tensor([True] + [False]*(self.size - 1))
            with open(target_path, 'rb') as f:
                tt = dill.load(f)
            target = tt.transform(df[['Стоимость']])
            self.data = np.concat([target, self.data[:, 1:]], axis=1)

    def get_mask(self):
        noise = torch.rand(self.size)
        ids = noise.argsort()
        mask = torch.concat([torch.ones(self.num_mask_features),
                             torch.zeros(self.size - self.num_mask_features)])
        return mask.gather(0, ids)

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        if self.num_mask_features is None:
            return torch.tensor(self.data[idx]), self.mask
        else:
            return torch.tensor(self.data[idx]), self.get_mask()
