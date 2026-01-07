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
                 target,
                 num_masks,
                 smooth=True
                 ) -> None:
        df = pd.read_csv(os.path.join(path, f"{dataset_type}.csv"))
        self.features = torch.as_tensor(data_transformer.transform(df))
        self.num_samples, self.num_features = self.features.shape

        if target == 'cat':
            num_bins = data_transformer.num_bins[0] if smooth else None
            self._target = self.get_target(self.features[:, 0], data_transformer.num_bins[0], num_bins)
            self._label = torch.as_tensor(df[['Стоимость']].values)
            self._mask = torch.zeros(self.num_features, dtype=torch.bool)
            self._mask[0] = True
        elif target == 'num':
            self._target = torch.as_tensor(data_transformer.transform(df[['Стоимость']], target=True))
            self._label = torch.as_tensor(df[['Стоимость']].values)
            self._mask = torch.zeros(self.num_features, dtype=torch.bool)
            self._mask[0] = True
        elif target == 'mask':
            num_bins = data_transformer.num_bins
            num_cats = data_transformer.num_cats
            offsets = data_transformer.offsets
            num_classes = sum(num_bins + num_cats)
            self._target = torch.cat(
                [
                    self.get_target(self.features[:, i], num_classes, num_bins[i], offsets[i: i + 2])
                    for i in range(len(num_bins))
                ] + [
                    self.get_target(self.features[:, i], num_classes)
                    for i in range(len(num_bins), len(num_bins) + len(num_cats))
                ],
                dim=1
            )
            # print(self._target[0])
        else:
            raise NotImplementedError()

        self.target = target
        self.num_masks = num_masks
        self.num_unmasks = self.num_features - self.num_masks

    def get_target(self, labels, num_classes, smooth_range=None, id_range=None):
        if smooth_range is None:
            target = F.one_hot(labels, num_classes=num_classes)
        else:
            smooth_range = smooth_range // 4
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
                    # if smooth_range != min(smooth_range, smooth_range // 2 - label + r) - max(0, smooth_range // 2 - label + l):
                    #     count += 1

                # return target
        return target.unsqueeze(1)

    def mask_target_label(self, idx):
        if self.target == 'mask':
            noise = torch.rand(self.num_features)
            ids = noise.argsort()
            mask = torch.cat(
                [torch.ones(self.num_masks),
                 torch.zeros(self.num_unmasks)]
            ).bool()
            mask = mask.gather(0, ids)
            return mask, self._target[idx][mask], self.features[idx][mask]
        else:
            return self._mask, self._target[idx], self._label[idx]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mask, target, label = self.mask_target_label(idx)
        return {'features': self.features[idx],
                'mask': mask,
                'target': target,
                'label': label}


if __name__ == '__main__':
    from configs.data_cfg import cfg
    ad = ApartmentDataset('train', cfg.path,
                          cfg.data_transformer, 'mask', 20)
    # t = ad[0]['features']
    # print(t.shape)
    # print(cfg.data_transformer.inverse_transform(t))  # , target=True))
    # print(t)
    labels = torch.randint(0, 12, [64,])
    ad.num_samples = 64
    print(labels)
    print(ad.get_target(labels, 64, 12, [0, 12]).squeeze()[torch.argwhere(labels == 0)])
