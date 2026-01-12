import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from accelerate import Accelerator
from data.apartment_dataset import ApartmentDataset
import models.transformers as models
from utils.utils import set_seed, get_scheduler, mape, accuracy


class Trainer:
    def __init__(self, cfg):
        set_seed(cfg.seed)

        self.cfg = cfg
        self._prepare_model(cfg.model_cfg)

        self.best_epoch = 0
        self.best_loss = float('inf')
        self.time_training = 0

        if cfg.target_type == 'mask':
            cfg.data_cfg.mask_ratio = cfg.mask_ratio
            cfg.data_cfg.offsets = self.model.embed.offsets.cpu().numpy()
            self.best_metric = float('-inf')
        elif cfg.target_type in ('cat', 'num'):
            self.best_metric = float('inf')
        else:
            raise NotImplementedError()

        self._prepare_data(cfg.data_cfg)

        self.criterion = getattr(nn, self.cfg.loss)(**self.cfg.loss_args)
        self.optimizer = self.model.configure_optimizer(self.cfg.lr, self.cfg.weight_decay,
                                                        self.cfg.get('lr_decay_by_block'))
        self.scheduler = get_scheduler(self.optimizer, len(self.train_dataloader) * self.cfg.num_epoch,
                                       self.cfg.lr_decay, self.cfg.lr, self.cfg.lr_decay_factor)

        self.accelerator = Accelerator(**cfg.accelerator_args)
        (
            self.model, self.optimizer, self.train_dataloader,
            self.val_dataloader, self.scheduler
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader,
            self.val_dataloader, self.scheduler
        )
        print(self.accelerator.device)

    def _prepare_data(self, data_cfg):
        self.data_transformer = data_cfg.data_transformer

        self.train_data = ApartmentDataset("train", **data_cfg)
        self.valid_data = ApartmentDataset('valid', **data_cfg)

        kwargs = {'batch_size': self.cfg.batch_size}
        if torch.cuda.is_available():
            kwargs['num_workers'] = 4
            kwargs['pin_memory'] = True
        self.train_dataloader = DataLoader(self.train_data, shuffle=True, **kwargs)
        self.val_dataloader = DataLoader(self.valid_data, shuffle=False, **kwargs)

    def _prepare_model(self, model_cfg):
        self.model = getattr(models, self.cfg.model)(**model_cfg)
        load_path = self.cfg.get('load_pretrained')
        if load_path is not None:
            self.load_model(load_path, strict=False)
            print('load_pretrained')

    def metric(self, pred, label):
        pred = pred.cpu()
        label = label.cpu()
        if self.cfg.target_type == 'mask':
            return accuracy(pred, label)
        elif self.cfg.target_type in ('num', 'cat'):
            # print(pred.shape)
            # print(label.shape)
            pred = torch.as_tensor(
                self.data_transformer.inverse_transform(pred, target=self.cfg.target_type)
            )
            return mape(pred, label)
        else:
            raise NotImplementedError()

    def save_model(self, **kwargs):
        os.makedirs(self.cfg.exp_dir, exist_ok=True)
        save_path = os.path.join(self.cfg.exp_dir, f"{self.cfg.model}.pt")
        torch.save(self.model.state_dict(**kwargs), save_path)

    def load_model(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = os.path.join(self.cfg.exp_dir, f"{self.cfg.model}.pt")
        self.model.load_state_dict(torch.load(load_path), **kwargs)

    def make_step(self, batch, update_model=True):
        with self.accelerator.autocast():
            if self.cfg.target_type == 'mask':
                pred = self.model(batch['features'], batch['mask'])
                pred = pred.reshape(batch['target'].shape).transpose(1, 2)
                batch['target'] = batch['target'].transpose(1, 2)
            else:
                pred = self.model(batch['features'])

            # print('pred', pred.shape)
            # print('target', batch['target'].shape)
            loss = self.criterion(pred, batch['target'])
            # loss = logcosh_loss(pred, batch['target'])

        if update_model:
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

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
            batch_len = len(batch['label'])  # * self.num_masks
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
            batch_len = len(batch['label'])  # * self.num_masks
            total_samples += batch_len
            total_loss += loss * batch_len
            total_metric += self.metric(pred, batch['label']) * batch_len

        total_loss /= total_samples
        total_metric /= total_samples
        self._print('valid', total_loss, total_metric, time.time() - t)

        if (
                (self.cfg.target_type in ('cat', 'num') and total_metric < self.best_metric) or
                (self.cfg.target_type == 'mask' and total_loss < self.best_loss)
        ):
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
    # from configs.pretrain_cfg import cfg

    trainer = Trainer(cfg)
    # trainer.overfitting_on_batch()
    trainer.fit()

"""
load_pretrained
cpu
Epoch 1/256
train loss: 1.2073 - metric: 0.623 - time: 1.5 (1.5) 
valid loss: 1.1772 - metric: 0.610 - time: 1.5 (0.1) 
best
Epoch 2/256
train loss: 1.1330 - metric: 0.570 - time: 2.9 (1.4) 
valid loss: 1.0580 - metric: 0.522 - time: 2.9 (0.1) 
best
Epoch 3/256
train loss: 1.0288 - metric: 0.488 - time: 4.3 (1.4) 
valid loss: 0.9517 - metric: 0.448 - time: 4.3 (0.1) 
best
Epoch 4/256
train loss: 0.9226 - metric: 0.440 - time: 5.7 (1.4) 
valid loss: 0.8264 - metric: 0.427 - time: 5.7 (0.1) 
best
Epoch 5/256
train loss: 0.7961 - metric: 0.419 - time: 7.1 (1.4) 
valid loss: 0.6816 - metric: 0.377 - time: 7.1 (0.1) 
best
Epoch 6/256
train loss: 0.6378 - metric: 0.350 - time: 8.5 (1.5) 
valid loss: 0.5064 - metric: 0.303 - time: 8.5 (0.1) 
best
Epoch 7/256
train loss: 0.4723 - metric: 0.294 - time: 10.0 (1.4) 
valid loss: 0.3524 - metric: 0.247 - time: 10.0 (0.1) 
best
Epoch 8/256
train loss: 0.3446 - metric: 0.245 - time: 11.4 (1.4) 
valid loss: 0.2612 - metric: 0.208 - time: 11.4 (0.1) 
best
Epoch 9/256
train loss: 0.2649 - metric: 0.212 - time: 12.8 (1.4) 
valid loss: 0.1951 - metric: 0.178 - time: 12.8 (0.1) 
best
Epoch 10/256
train loss: 0.2083 - metric: 0.185 - time: 14.2 (1.4) 
valid loss: 0.1517 - metric: 0.161 - time: 14.2 (0.1) 
best
Epoch 11/256
train loss: 0.1668 - metric: 0.165 - time: 15.7 (1.4) 
valid loss: 0.1214 - metric: 0.143 - time: 15.7 (0.1) 
best
Epoch 12/256
train loss: 0.1384 - metric: 0.149 - time: 17.0 (1.4) 
valid loss: 0.0995 - metric: 0.124 - time: 17.0 (0.1) 
best
Epoch 13/256
train loss: 0.1188 - metric: 0.137 - time: 18.5 (1.4) 
valid loss: 0.0858 - metric: 0.118 - time: 18.5 (0.1) 
best
Epoch 14/256
train loss: 0.1035 - metric: 0.128 - time: 19.9 (1.4) 
valid loss: 0.0757 - metric: 0.108 - time: 19.9 (0.1) 
best
Epoch 15/256
train loss: 0.0922 - metric: 0.121 - time: 21.3 (1.5) 
valid loss: 0.0680 - metric: 0.102 - time: 21.3 (0.1) 
best
Epoch 16/256
train loss: 0.0844 - metric: 0.115 - time: 22.9 (1.5) 
valid loss: 0.0631 - metric: 0.099 - time: 22.9 (0.1) 
best
Epoch 17/256
train loss: 0.0785 - metric: 0.110 - time: 24.3 (1.4) 
valid loss: 0.0589 - metric: 0.097 - time: 24.3 (0.1) 
best
Epoch 18/256
train loss: 0.0741 - metric: 0.107 - time: 25.7 (1.5) 
valid loss: 0.0557 - metric: 0.092 - time: 25.7 (0.1) 
best
Epoch 19/256
train loss: 0.0707 - metric: 0.105 - time: 27.2 (1.5) 
valid loss: 0.0532 - metric: 0.090 - time: 27.2 (0.1) 
best
Epoch 20/256
train loss: 0.0670 - metric: 0.102 - time: 28.6 (1.4) 
valid loss: 0.0530 - metric: 0.093 - time: 28.6 (0.1) 
Epoch 21/256

"""
