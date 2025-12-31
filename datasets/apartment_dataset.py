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
                 ) -> None:
        df = pd.read_csv(data_path)
        with open(ct_path, 'rb') as f:
            ct = dill.load(f)
        self.data = ct.transform(df)
        if target_path is not None:
            self.mask = torch.tensor([True] + [False]*(len(df) - 1))
            with open(target_path, 'rb') as f:
                tt = dill.load(f)
            target = tt.transform(df[['Стоимость']])
            self.data = np.concat([target, self.data[:, 1:]], axis=1)

    def get_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.data[idx])
