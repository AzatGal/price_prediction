import os
import time

from easydict import EasyDict

# from models.ensembles import PricePredEnsemble
import models.tabm as tabm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from accelerate import Accelerator
from data.apartment_dataset import ApartmentDataset
from models.ensembles import PricePredEnsemble
from utils.utils import set_seed, get_scheduler, mape


class TabmTrainer:
    def __init__(self, cfg):
        set_seed(cfg.seed)
        self.cfg = cfg
        self._prepare_data(cfg.data_cfg)
        self._prepare_model(cfg.model_cfg)
        self.accelerator = Accelerator(**self.cfg.accelerator_args)
        (
            self.model, self.optimizer, self.train_dataloader,
            self.val_dataloader, self.scheduler
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader,
            self.val_dataloader, self.scheduler
        )
        print(self.accelerator.device)

        self.best_epoch = 0
        self.best_metric = float('inf')
        self.best_loss = float('inf')
        self.time_training = 0

    def _prepare_data(self, data_cfg):
        self.data_transformer = data_cfg.data_transformer
        # self.data_transformer.apply_offsets = False
        self.num_masks = self.cfg.get('num_masks')
        self.train_data = ApartmentDataset("train", data_cfg.path, self.data_transformer,
                                           self.cfg.target, self.num_masks)
        self.val_data = ApartmentDataset('valid', data_cfg.path, self.data_transformer,
                                         self.cfg.target, self.num_masks)
        kwargs = {'batch_size': self.cfg.batch_size}
        if torch.cuda.is_available():
            kwargs['num_workers'] = 4
            kwargs['pin_memory'] = True
        self.train_dataloader = DataLoader(self.train_data, shuffle=True, **kwargs)
        self.val_dataloader = DataLoader(self.val_data, shuffle=False, **kwargs)

    def _prepare_model(self, model_cfg):
        # self.model = tabm.TabM.make(**model_cfg)
        # d_in = 24
        # d = 512
        # d_out = 1
        # k = 32
        #
        # # Any MLP-like backbone can be used.
        # backbone = tabm.MLPBackbone(
        #     d_in=d_in, n_blocks=2, d_block=d, dropout=0.1
        # )
        # self.model = nn.Sequential(
        #     tabm.EnsembleView(k=k),
        #     tabm.ElementwiseAffine((k, d_in), bias=False, scaling_init='normal'),
        #     backbone,
        #     tabm.LinearEnsemble(d, d_out, k=k),
        # )
        self.model = PricePredEnsemble(**model_cfg)
        self.criterion = getattr(nn, self.cfg.loss)(**self.cfg.loss_args)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr,
        #                                    weight_decay=self.cfg.weight_decay)
        self.optimizer = self.model.configure_optimizer(self.cfg.lr, self.cfg.weight_decay,
                                                        self.cfg.get('lr_decay_by_block'))
        self.scheduler = get_scheduler(self.optimizer, len(self.train_dataloader) * self.cfg.num_epoch,
                                       self.cfg.decay, self.cfg.lr, self.cfg.lr_decay_factor)

    def metric(self, pred, label):
        pred = pred.cpu()
        label = label.cpu()
        # print('pred', pred.shape)
        # print('label', label.shape)
        pred = torch.as_tensor(
            self.data_transformer.inverse_transform(pred, target=self.cfg.target)
        )
        return mape(pred, label)

    def save_model(self):
        os.makedirs(self.cfg.exp_dir, exist_ok=True)
        # save_path = os.path.join(self.cfg.exp_dir, "tabm.pt")  # 'ens.pt')  #
        save_path = os.path.join(self.cfg.exp_dir, 'ens.pt')  #
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path=None):
        if load_path is None:
            # load_path = os.path.join(self.cfg.exp_dir, "tabm.pt")  # 'ens.pt')  # "tabm.pt")
            load_path = os.path.join(self.cfg.exp_dir, 'ens.pt')  # "tabm.pt")
        self.model.load_state_dict(torch.load(load_path))

    def make_step(self, batch, update_model=True):
        with self.accelerator.autocast():
            # pred = self.model(x_cat=batch['features'][:, 1:])
            pred = self.model(batch['features'], batch['mask'])
            pred = pred.squeeze(-1)
            # print(pred.shape)
            if self.cfg.target == 'cat':
                target = batch['target'].unsqueeze(1).expand(-1, pred.size(1), -1)
                pred = pred.transpose(1, 2)
                target = target.transpose(1, 2)
            elif self.cfg.target == 'num':
                target = batch['target'].expand(-1, pred.size(1))
            # print(pred.shape)
            # print(target.shape)
            loss = self.criterion(pred, target)

        if update_model:
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        if self.cfg.target == 'cat':
            pred = pred.sum(2)
        elif self.cfg.target == 'num':
            pred = pred.mean(1)
        else:
            raise NotImplementedError()
        return loss.item(), pred.detach()

    def _print(self, s, l, m, t):
        print(f'{s} loss: {l:.4f} - metric: {m:.3f} - time: {self.time_training:.1f} ({t:.1f}) ')

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_metric = 0
        total_samples = 0

        t = time.time()
        for i, batch in enumerate(self.train_dataloader):
            loss, pred = self.make_step(batch)
            batch_len = len(batch['label']) * self.num_masks
            total_samples += batch_len
            total_loss += loss * batch_len
            total_metric += self.metric(pred, batch['label']) * batch_len

        t = time.time() - t
        total_loss /= total_samples
        total_metric /= total_samples
        self.time_training += t

        self._print('train', total_loss, total_metric, t)

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_metric = 0
        total_samples = 0

        t = time.time()
        for i, batch in enumerate(self.val_dataloader):
            loss, pred = self.make_step(batch, False)
            batch_len = len(batch['label']) * self.num_masks
            total_samples += batch_len
            total_loss += loss * batch_len
            total_metric += self.metric(pred, batch['label']) * batch_len

        t = time.time() - t
        total_loss /= total_samples
        total_metric /= total_samples
        self.time_training += t

        self._print('valid', total_loss, total_metric, t)
        if total_metric < self.best_metric:
            print('best')
            self.save_model()
            self.best_metric = total_metric
            self.best_loss = total_loss
            self.best_epoch = epoch

    def fit(self):
        for epoch in range(self.cfg.num_epoch):
            epoch = epoch + 1
            print(f"Epoch {epoch}/{self.cfg.num_epoch}")
            self.train_epoch()
            self.evaluate(epoch)
        print(f"\nbest epoch: {self.best_epoch} - metric: {self.best_metric} - loss: {self.best_loss}")

    def overfitting_on_batch(self, max_step=1000):
        batch = next(iter(self.train_dataloader))
        for step in range(max_step):
            loss, output = self.make_step(batch, update_model=True)
            if step % 100 == 0:
                print(f'[{step}]: loss - {loss:.4f}')


if __name__ == "__main__":
    from configs.train_cfg import cfg

    cfg.weight_decay = 3e-4
    #
    # cfg.model_cfg = EasyDict(
    #     cat_cardinalities=cfg.model_cfg.num_embed_features[1:],
    #     d_out=1,
    #     # arch_type='tabm-mini'
    #     # d_in,
    #     # n_blocks,
    #     # d_block,
    #     # dropout=0.1,
    #     # activation='ReLU',
    #     # k=32
    # )
    model_cfg = EasyDict()

    model_cfg.k = 12
    model_cfg.embed_dim = 12  # 24
    model_cfg.num_heads = 2
    model_cfg.num_blocks = 1  # 3
    model_cfg.act = 'SiLU'  # SiLU
    model_cfg.num_embed_features = (cfg.data_cfg.data_transformer.num_bins +
                                    cfg.data_cfg.data_transformer.num_cats)
    model_cfg.pred_dim = 1  #
    model_cfg.attn_dropout = 0.05
    model_cfg.mlp_dropout = 0.1
    model_cfg.dropout = 0.1
    model_cfg.compression_factor = 0.15
    model_cfg.compression = 'KV'  # Head KV Layer
    model_cfg.mlp_dim_factor = 5 / 3
    model_cfg.norm = 'LayerNorm'
    model_cfg.log_softmax = False  # False True

    cfg.model_cfg = model_cfg

    trainer = TabmTrainer(cfg)
    # trainer.overfitting_on_batch()
    trainer.fit()

"""
cpu
Epoch 1/125
train loss: 4.4923 - metric: 0.543 - time: 21.0 (21.0) 
valid loss: 3.9489 - metric: 0.257 - time: 22.0 (1.0) 
best
Epoch 2/125
train loss: 3.5333 - metric: 0.153 - time: 45.7 (23.7) 
valid loss: 3.2604 - metric: 0.119 - time: 46.9 (1.2) 
best
Epoch 3/125
train loss: 3.3376 - metric: 0.122 - time: 70.8 (23.9) 
valid loss: 3.1580 - metric: 0.116 - time: 72.0 (1.2) 
best
Epoch 4/125
train loss: 3.2289 - metric: 0.111 - time: 96.1 (24.1) 
valid loss: 3.0216 - metric: 0.092 - time: 97.3 (1.2) 
best
Epoch 5/125
train loss: 3.1364 - metric: 0.102 - time: 121.3 (24.0) 
valid loss: 3.0314 - metric: 0.094 - time: 122.5 (1.2) 
Epoch 6/125
train loss: 3.0893 - metric: 0.098 - time: 146.3 (23.8) 
valid loss: 3.0127 - metric: 0.089 - time: 147.5 (1.2) 
best
Epoch 7/125
train loss: 3.0510 - metric: 0.094 - time: 171.6 (24.1) 
valid loss: 3.0213 - metric: 0.105 - time: 172.7 (1.2) 
Epoch 8/125
train loss: 2.9978 - metric: 0.089 - time: 196.8 (24.1) 
valid loss: 2.9651 - metric: 0.106 - time: 198.0 (1.2) 
Epoch 9/125
train loss: 2.9590 - metric: 0.087 - time: 222.0 (23.9) 
valid loss: 2.8622 - metric: 0.088 - time: 223.2 (1.2) 
best
Epoch 10/125
train loss: 2.9293 - metric: 0.085 - time: 247.1 (23.9) 
valid loss: 2.9927 - metric: 0.094 - time: 248.3 (1.2) 
Epoch 11/125
train loss: 2.9038 - metric: 0.083 - time: 272.3 (24.1) 
valid loss: 2.8241 - metric: 0.083 - time: 273.5 (1.2) 
best
Epoch 12/125
train loss: 2.8786 - metric: 0.081 - time: 297.5 (24.0) 
valid loss: 2.8797 - metric: 0.090 - time: 298.9 (1.4) 
Epoch 13/125
train loss: 2.8597 - metric: 0.079 - time: 326.2 (27.3) 
valid loss: 2.8930 - metric: 0.091 - time: 327.7 (1.5) 
Epoch 14/125
train loss: 2.8378 - metric: 0.077 - time: 357.1 (29.4) 
valid loss: 2.8477 - metric: 0.081 - time: 358.3 (1.2) 
best
Epoch 15/125
train loss: 2.8226 - metric: 0.076 - time: 384.7 (26.4) 
valid loss: 2.8437 - metric: 0.081 - time: 386.0 (1.2) 
Epoch 16/125
train loss: 2.8068 - metric: 0.075 - time: 415.2 (29.2) 
valid loss: 2.7880 - metric: 0.073 - time: 416.6 (1.4) 
best
Epoch 17/125

cpu
Epoch 1/125
train loss: 0.3667 - metric: 0.192 - time: 35.3 (35.3) 
valid loss: 0.1840 - metric: 0.087 - time: 35.3 (1.2) 
best
Epoch 2/125
train loss: 0.1881 - metric: 0.085 - time: 70.0 (34.8) 
valid loss: 0.1720 - metric: 0.079 - time: 70.0 (1.3) 
best
Epoch 3/125
train loss: 0.1736 - metric: 0.077 - time: 106.7 (36.6) 
valid loss: 0.1697 - metric: 0.079 - time: 106.7 (1.2) 
best
Epoch 4/125
train loss: 0.1669 - metric: 0.073 - time: 140.8 (34.1) 
valid loss: 0.1547 - metric: 0.070 - time: 140.8 (1.2) 
best
Epoch 5/125
train loss: 0.1628 - metric: 0.069 - time: 174.6 (33.8) 
valid loss: 0.1598 - metric: 0.068 - time: 174.6 (1.2) 
best
Epoch 6/125
train loss: 0.1593 - metric: 0.066 - time: 208.8 (34.2) 
valid loss: 0.1537 - metric: 0.066 - time: 208.8 (1.2) 
best
Epoch 7/125
train loss: 0.1541 - metric: 0.063 - time: 244.8 (36.0) 
valid loss: 0.1448 - metric: 0.066 - time: 244.8 (1.3) 
best
Epoch 8/125
train loss: 0.1462 - metric: 0.059 - time: 280.0 (35.3) 
valid loss: 0.1379 - metric: 0.062 - time: 280.0 (1.2) 
best
Epoch 9/125
train loss: 0.1405 - metric: 0.056 - time: 314.6 (34.5) 
valid loss: 0.1322 - metric: 0.060 - time: 314.6 (1.1) 
best
Epoch 10/125
train loss: 0.1345 - metric: 0.053 - time: 349.1 (34.6) 
valid loss: 0.1411 - metric: 0.064 - time: 349.1 (1.1) 
Epoch 11/125
train loss: 0.1305 - metric: 0.051 - time: 382.9 (33.8) 
valid loss: 0.1243 - metric: 0.054 - time: 382.9 (1.3) 
best
Epoch 12/125
train loss: 0.1263 - metric: 0.049 - time: 417.3 (34.4) 
valid loss: 0.1244 - metric: 0.058 - time: 417.3 (1.1) 
Epoch 13/125
train loss: 0.1227 - metric: 0.047 - time: 451.9 (34.6) 
valid loss: 0.1208 - metric: 0.054 - time: 451.9 (1.2) 
best
Epoch 14/125
train loss: 0.1198 - metric: 0.045 - time: 486.5 (34.6) 
valid loss: 0.1193 - metric: 0.053 - time: 486.5 (1.2) 
best
Epoch 15/125
train loss: 0.1173 - metric: 0.044 - time: 520.6 (34.1) 
valid loss: 0.1184 - metric: 0.054 - time: 520.6 (1.1) 
Epoch 16/125
train loss: 0.1146 - metric: 0.042 - time: 554.5 (33.9) 
valid loss: 0.1189 - metric: 0.052 - time: 554.5 (1.2) 
best
Epoch 17/125
train loss: 0.1128 - metric: 0.041 - time: 588.9 (34.4) 
valid loss: 0.1202 - metric: 0.055 - time: 588.9 (1.1) 
Epoch 18/125
train loss: 0.1107 - metric: 0.039 - time: 625.2 (36.3) 
valid loss: 0.1184 - metric: 0.054 - time: 625.2 (1.2) 
Epoch 19/125
train loss: 0.1088 - metric: 0.039 - time: 659.3 (34.2) 
valid loss: 0.1197 - metric: 0.057 - time: 659.3 (1.2) 
Epoch 20/125
train loss: 0.1074 - metric: 0.038 - time: 693.4 (34.1) 
valid loss: 0.1132 - metric: 0.048 - time: 693.4 (1.2) 
best
Epoch 21/125
train loss: 0.1057 - metric: 0.037 - time: 727.7 (34.3) 
valid loss: 0.1162 - metric: 0.054 - time: 727.7 (1.2) 
Epoch 22/125
train loss: 0.1041 - metric: 0.036 - time: 762.3 (34.6) 
valid loss: 0.1127 - metric: 0.049 - time: 762.3 (1.2) 
Epoch 23/125
train loss: 0.1027 - metric: 0.035 - time: 796.7 (34.5) 
valid loss: 0.1134 - metric: 0.049 - time: 796.7 (1.1) 
Epoch 24/125
train loss: 0.1014 - metric: 0.034 - time: 831.3 (34.5) 
valid loss: 0.1128 - metric: 0.048 - time: 831.3 (1.1) 
best
Epoch 25/125
train loss: 0.1003 - metric: 0.033 - time: 865.5 (34.3) 
valid loss: 0.1095 - metric: 0.047 - time: 865.5 (1.2) 
best
Epoch 26/125
train loss: 0.0990 - metric: 0.033 - time: 899.6 (34.1) 
valid loss: 0.1100 - metric: 0.048 - time: 899.6 (1.2) 
Epoch 27/125
train loss: 0.0983 - metric: 0.032 - time: 933.8 (34.2) 
valid loss: 0.1112 - metric: 0.050 - time: 933.8 (1.2) 
Epoch 28/125
train loss: 0.0970 - metric: 0.031 - time: 967.6 (33.8) 
valid loss: 0.1104 - metric: 0.049 - time: 967.6 (1.1) 
Epoch 29/125
train loss: 0.0961 - metric: 0.031 - time: 1001.5 (33.9) 
valid loss: 0.1092 - metric: 0.047 - time: 1001.5 (1.2) 
best
Epoch 30/125
train loss: 0.0949 - metric: 0.030 - time: 1035.4 (33.9) 
valid loss: 0.1114 - metric: 0.048 - time: 1035.4 (1.3) 
Epoch 31/125
train loss: 0.0941 - metric: 0.030 - time: 1069.6 (34.2) 
valid loss: 0.1076 - metric: 0.046 - time: 1069.6 (1.2) 
best
Epoch 32/125
train loss: 0.0932 - metric: 0.029 - time: 1104.7 (35.1) 
valid loss: 0.1073 - metric: 0.047 - time: 1104.7 (1.1) 
Epoch 33/125
train loss: 0.0923 - metric: 0.029 - time: 1138.7 (34.0) 
valid loss: 0.1087 - metric: 0.047 - time: 1138.7 (1.2) 
Epoch 34/125
train loss: 0.0915 - metric: 0.028 - time: 1172.6 (34.0) 
valid loss: 0.1056 - metric: 0.046 - time: 1172.6 (1.2) 
best
Epoch 35/125
train loss: 0.0907 - metric: 0.028 - time: 1207.0 (34.3) 
valid loss: 0.1053 - metric: 0.046 - time: 1207.0 (1.2) 
Epoch 36/125
train loss: 0.0900 - metric: 0.028 - time: 1240.9 (33.9) 
valid loss: 0.1059 - metric: 0.047 - time: 1240.9 (1.1) 
Epoch 37/125
train loss: 0.0892 - metric: 0.027 - time: 1274.7 (33.8) 
valid loss: 0.1062 - metric: 0.047 - time: 1274.7 (1.1) 
Epoch 38/125
train loss: 0.0885 - metric: 0.027 - time: 1308.7 (34.1) 
valid loss: 0.1057 - metric: 0.047 - time: 1308.7 (1.2) 
Epoch 39/125
train loss: 0.0877 - metric: 0.026 - time: 1342.9 (34.1) 
valid loss: 0.1051 - metric: 0.045 - time: 1342.9 (1.2) 
best
Epoch 40/125
train loss: 0.0870 - metric: 0.026 - time: 1376.9 (34.1) 
valid loss: 0.1039 - metric: 0.046 - time: 1376.9 (1.1) 
Epoch 41/125
train loss: 0.0866 - metric: 0.026 - time: 1410.9 (33.9) 
valid loss: 0.1042 - metric: 0.046 - time: 1410.9 (1.1) 
Epoch 42/125
train loss: 0.0857 - metric: 0.025 - time: 1444.9 (34.0) 
valid loss: 0.1046 - metric: 0.046 - time: 1444.9 (1.2) 
Epoch 43/125
train loss: 0.0851 - metric: 0.025 - time: 1478.9 (34.0) 
valid loss: 0.1044 - metric: 0.047 - time: 1478.9 (1.2) 
Epoch 44/125
train loss: 0.0847 - metric: 0.025 - time: 1513.8 (34.9) 
valid loss: 0.1040 - metric: 0.046 - time: 1513.8 (1.4) 
Epoch 45/125
train loss: 0.0839 - metric: 0.025 - time: 1548.3 (34.6) 
valid loss: 0.1042 - metric: 0.045 - time: 1548.3 (1.2) 
best
Epoch 46/125
train loss: 0.0831 - metric: 0.024 - time: 1582.3 (34.0) 
valid loss: 0.1029 - metric: 0.046 - time: 1582.3 (1.1) 
Epoch 47/125
train loss: 0.0825 - metric: 0.024 - time: 1617.6 (35.2) 
valid loss: 0.1057 - metric: 0.047 - time: 1617.6 (1.4) 
Epoch 48/125
train loss: 0.0819 - metric: 0.024 - time: 1652.5 (34.9) 
valid loss: 0.1020 - metric: 0.045 - time: 1652.5 (1.2) 
best
Epoch 49/125
train loss: 0.0814 - metric: 0.023 - time: 1693.8 (41.3) 
valid loss: 0.1026 - metric: 0.047 - time: 1693.8 (1.5) 
Epoch 50/125
train loss: 0.0807 - metric: 0.023 - time: 1734.9 (41.1) 
valid loss: 0.1022 - metric: 0.046 - time: 1734.9 (1.3) 
Epoch 51/125
train loss: 0.0802 - metric: 0.023 - time: 1775.4 (40.5) 
valid loss: 0.1025 - metric: 0.046 - time: 1775.4 (1.4) 
Epoch 52/125
train loss: 0.0795 - metric: 0.022 - time: 1815.7 (40.2) 
valid loss: 0.1031 - metric: 0.046 - time: 1815.7 (1.5) 
Epoch 53/125
train loss: 0.0790 - metric: 0.022 - time: 1858.0 (42.3) 
valid loss: 0.1015 - metric: 0.045 - time: 1858.0 (2.6) 
Epoch 54/125
train loss: 0.0784 - metric: 0.022 - time: 1900.1 (42.1) 
valid loss: 0.1007 - metric: 0.044 - time: 1900.1 (2.7) 
best
Epoch 55/125
train loss: 0.0780 - metric: 0.022 - time: 1944.0 (43.9) 
valid loss: 0.1004 - metric: 0.044 - time: 1944.0 (1.3) 
best
Epoch 56/125
train loss: 0.0773 - metric: 0.021 - time: 1987.5 (43.6) 
valid loss: 0.1015 - metric: 0.045 - time: 1987.5 (1.3) 
Epoch 57/125
train loss: 0.0767 - metric: 0.021 - time: 2029.3 (41.8) 
valid loss: 0.0988 - metric: 0.045 - time: 2029.3 (1.4) 
Epoch 58/125
train loss: 0.0763 - metric: 0.021 - time: 2072.2 (42.9) 
valid loss: 0.0995 - metric: 0.045 - time: 2072.2 (1.4) 
Epoch 59/125
train loss: 0.0757 - metric: 0.021 - time: 2114.3 (42.1) 
valid loss: 0.0985 - metric: 0.044 - time: 2114.3 (1.4) 
Epoch 60/125
train loss: 0.0753 - metric: 0.021 - time: 2157.8 (43.4) 
valid loss: 0.0999 - metric: 0.044 - time: 2157.8 (1.3) 
Epoch 61/125
train loss: 0.0748 - metric: 0.020 - time: 2197.4 (39.6) 
valid loss: 0.0982 - metric: 0.043 - time: 2197.4 (1.5) 
best
Epoch 62/125
train loss: 0.0744 - metric: 0.020 - time: 2240.6 (43.3) 
valid loss: 0.0987 - metric: 0.044 - time: 2240.6 (1.4) 
Epoch 63/125
train loss: 0.0737 - metric: 0.020 - time: 2282.2 (41.5) 
valid loss: 0.0983 - metric: 0.044 - time: 2282.2 (1.4) 
Epoch 64/125
train loss: 0.0732 - metric: 0.020 - time: 2324.9 (42.7) 
valid loss: 0.0982 - metric: 0.045 - time: 2324.9 (1.7) 
Epoch 65/125
train loss: 0.0727 - metric: 0.019 - time: 2366.5 (41.6) 
valid loss: 0.0983 - metric: 0.045 - time: 2366.5 (1.3) 
Epoch 66/125
train loss: 0.0723 - metric: 0.019 - time: 2407.3 (40.8) 
valid loss: 0.0976 - metric: 0.044 - time: 2407.3 (1.4) 
Epoch 67/125
train loss: 0.0719 - metric: 0.019 - time: 2448.5 (41.1) 
valid loss: 0.0976 - metric: 0.044 - time: 2448.5 (1.4) 
Epoch 68/125
train loss: 0.0715 - metric: 0.019 - time: 2488.1 (39.7) 
valid loss: 0.0971 - metric: 0.043 - time: 2488.1 (1.3) 
best
Epoch 69/125
train loss: 0.0708 - metric: 0.019 - time: 2528.3 (40.2) 
valid loss: 0.0971 - metric: 0.044 - time: 2528.3 (1.4) 
Epoch 70/125
train loss: 0.0705 - metric: 0.019 - time: 2568.2 (39.8) 
valid loss: 0.0987 - metric: 0.045 - time: 2568.2 (1.3) 
Epoch 71/125
train loss: 0.0700 - metric: 0.018 - time: 2609.5 (41.3) 
valid loss: 0.0969 - metric: 0.044 - time: 2609.5 (1.6) 
Epoch 72/125
train loss: 0.0695 - metric: 0.018 - time: 2650.6 (41.1) 
valid loss: 0.0967 - metric: 0.044 - time: 2650.6 (1.3) 
Epoch 73/125
train loss: 0.0691 - metric: 0.018 - time: 2692.5 (41.9) 
valid loss: 0.0966 - metric: 0.044 - time: 2692.5 (1.3) 
Epoch 74/125
train loss: 0.0687 - metric: 0.018 - time: 2733.5 (41.0) 
valid loss: 0.0965 - metric: 0.044 - time: 2733.5 (1.3) 
Epoch 75/125
train loss: 0.0682 - metric: 0.018 - time: 2773.5 (40.0) 
valid loss: 0.0953 - metric: 0.044 - time: 2773.5 (1.4) 
Epoch 76/125
train loss: 0.0679 - metric: 0.017 - time: 2813.5 (40.0) 
valid loss: 0.0955 - metric: 0.043 - time: 2813.5 (1.3) 
best
Epoch 77/125
train loss: 0.0674 - metric: 0.017 - time: 2853.5 (40.0) 
valid loss: 0.0965 - metric: 0.045 - time: 2853.5 (1.3) 
Epoch 78/125
"""
