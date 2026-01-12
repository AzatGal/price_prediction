import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ApartmentDataset(Dataset):
    def __init__(self,
                 dataset_type,
                 path,
                 data_transformer,
                 target_type,
                 mask_first_token=False,
                 smooth=True,
                 **kwargs
                 ) -> None:
        df = pd.read_csv(os.path.join(path, f"{dataset_type}.csv"))
        features = torch.as_tensor(
            data_transformer.transform(df)
        )

        self.mask_first_token = mask_first_token
        if self.mask_first_token:
            self.features = features
            self.mask = torch.zeros(self.features.size(1), dtype=torch.bool)
            self.mask[0] = True
        else:
            self.features = features[:, 1:]

        self.num_samples, self.num_features = self.features.shape

        if target_type == 'cat':
            num_bins = data_transformer.num_bins[0] if smooth else None
            self.target = self.get_target(features, data_transformer.num_bins[0],
                                          num_bins, [0, num_bins]).squeeze()
            self.label = torch.as_tensor(df[['Стоимость']].values)
        elif target_type == 'num':
            self.target = torch.as_tensor(
                data_transformer.transform(df[['Стоимость']], target=True)
            )
            self.label = torch.as_tensor(df[['Стоимость']].values)
        elif target_type == 'mask':
            num_mask = int(kwargs['mask_ratio'] * self.num_features)
            num_unmask = self.num_features - num_mask
            self.mask = torch.cat([torch.ones(num_mask),
                                   torch.zeros(num_unmask)]).bool()

            num_bins = data_transformer.num_bins[
                0 if self.mask_first_token else 1:
            ]
            num_cats = data_transformer.num_cats
            offsets = kwargs['offsets']
            # print(offsets)
            num_classes = sum(num_bins + num_cats)
            num_bins_len = len(num_bins)
            num_cats_len = len(num_cats)
            # print(num_bins_len, num_cats_len)

            self.target = torch.cat(
                [
                    self.get_target(self.features[:, i], num_classes, num_bins[i], offsets[i: i + 2])
                    for i in range(num_bins_len)
                ] + [
                    self.get_target(self.features[:, i], num_classes)
                    for i in range(num_bins_len, num_bins_len + num_cats_len)
                ],
                dim=1
            )
            # print(self.target)
        else:
            raise NotImplementedError()

        self.target = self.target.float()
        self.target_type = target_type

    def get_target(self, labels, num_classes, smooth_range=None, id_range=None):
        if smooth_range is None:
            target = F.one_hot(labels, num_classes=num_classes)
        else:
            smooth_range = smooth_range // 2
            smooth_range = smooth_range + smooth_range % 2 + 1
            if smooth_range < 5:
                target = F.one_hot(labels, num_classes=num_classes)
            else:
                values = F.softmax(
                    torch.signal.windows.gaussian(smooth_range) * 8,
                    dim=0
                )
                target = torch.zeros(self.num_samples, num_classes)
                for i in range(self.num_samples):
                    label = labels[i].item()
                    l = max(label - smooth_range // 2, id_range[0])
                    r = min(label + smooth_range // 2 + 1, id_range[1])
                    target[i, l: r] = values[
                        max(0, smooth_range // 2 - label + l):
                        min(smooth_range, smooth_range // 2 - label + r)
                    ]
        return target.unsqueeze(1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.target_type == 'mask':
            mask = self.mask.gather(0, torch.rand(self.num_features).argsort())
            # print(mask.shape)
            # print(self.target.shape)
            return {
                'features': self.features[idx],
                'mask': mask,
                'target': self.target[idx][mask],
                'label': self.features[idx][mask]
            }
        else:
            res = {
                'target': self.target[idx],
                'label': self.label[idx]
            }
            if self.mask_first_token:
                res['features'] = self.features[idx]
                res['mask'] = self.mask
            else:
                res['features'] = self.features[idx]
            return res


if __name__ == '__main__':
    from configs.data_cfg import cfg
    ad = ApartmentDataset('train', cfg.path,
                          cfg.data_transformer, 'num', 19)
    # print(cfg.data_transformer.offsets)
    t = ad[32]['features']
    print(t)
    # print(t.shape)
    # print(cfg.data_transformer.inverse_transform(t))  # , target=True))
    # print(t)
    # labels = torch.randint(0, 12, [64,])
    # ad.num_samples = 64
    # print(labels)
    # print(ad.get_target(labels, 64, 12, [0, 12]).squeeze()[torch.argwhere(labels == 0)])
