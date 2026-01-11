import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from accelerate import Accelerator
from data.apartment_dataset import ApartmentDataset
from models.transformers import MaskedTableModeling, PricePrediction, MaskedTableAutoencoder
from utils.utils import set_seed, get_scheduler, mape, accuracy, logcosh_loss


class Trainer:
    def __init__(self, cfg):
        set_seed(cfg.seed)
        self.cfg = cfg
        self._prepare_data(cfg.data_cfg)
        self._prepare_model(cfg.model_cfg)

        self.accelerator = Accelerator(**cfg.accelerator_args)
        (
            self.model, self.optimizer, self.train_dataloader,
            self.val_dataloader, self.scheduler
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader,
            self.val_dataloader, self.scheduler
        )
        print(self.accelerator.device)

        self.best_epoch = 0
        if self.cfg.target == 'mask':
            self.best_metric = float('-inf')
        elif self.cfg.target in ('cat', 'num'):
            self.best_metric = float('inf')
        else:
            raise NotImplementedError()
        self.best_loss = float('inf')
        self.time_training = 0

    def _prepare_data(self, data_cfg):
        self.data_transformer = data_cfg.data_transformer
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
        if self.cfg.model == 'MaskedTableModeling':
            self.model = MaskedTableModeling(**model_cfg)
        elif self.cfg.model == 'MaskedTableAutoencoder':
            self.model = MaskedTableAutoencoder(**model_cfg)
        elif self.cfg.model == 'PricePrediction':
            self.model = PricePrediction(**model_cfg)
        else:
            raise NotImplementedError()
        # print(self.model)
        load_path = self.cfg.get('load_pretrained')
        if load_path is not None:
            self.load_model(load_path)
            print('load_pretrained')
        self.criterion = getattr(nn, self.cfg.loss)(**self.cfg.loss_args)
        self.optimizer = self.model.configure_optimizer(self.cfg.lr, self.cfg.weight_decay,
                                                        self.cfg.get('lr_decay_by_block'))
        self.scheduler = get_scheduler(self.optimizer, len(self.train_dataloader) * self.cfg.num_epoch,
                                       self.cfg.decay, self.cfg.lr, self.cfg.lr_decay_factor)

    def metric(self, pred, label):
        pred = pred.cpu()
        label = label.cpu()
        if self.cfg.target == 'mask':
            return accuracy(pred, label)
        elif self.cfg.target in ('num', 'cat'):
            # print(pred.shape)
            # print(label.shape)
            pred = torch.as_tensor(
                self.data_transformer.inverse_transform(pred, target=self.cfg.target)
            )
            return mape(pred, label)
        else:
            raise NotImplementedError()

    def save_model(self):
        os.makedirs(self.cfg.exp_dir, exist_ok=True)
        save_path = os.path.join(self.cfg.exp_dir, f"{self.cfg.model}.pt")
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path=None):
        if load_path is None:
            load_path = os.path.join(self.cfg.exp_dir, f"{self.cfg.model}.pt")
        self.model.load_state_dict(torch.load(load_path), strict=False)

    def make_step(self, batch, update_model=True):
        with self.accelerator.autocast():
            pred = self.model(batch['features'], batch['mask'])
            if self.cfg.target == 'mask':
                pred = pred.transpose(1, 2)
                batch['target'] = batch['target'].transpose(1, 2)
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

        total_loss /= total_samples
        total_metric /= total_samples
        self._print('valid', total_loss, total_metric, time.time() - t)

        if (
                (self.cfg.target in ('cat', 'num') and total_metric < self.best_metric) or
                (self.cfg.target == 'mask' and total_metric > self.best_metric)
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

    # cfg.lr = 1e-4
    trainer = Trainer(cfg)
    # trainer.load_model()
    # trainer.overfitting_on_batch()
    trainer.fit()

"""
cpu
Epoch 1/125
train loss: 1.1554 - metric: 0.503 - time: 7.4 (7.4) 
valid loss: 0.9043 - metric: 0.436 - time: 7.4 (0.4) 
best
Epoch 2/125
train loss: 0.6577 - metric: 0.358 - time: 13.5 (6.1) 
valid loss: 0.2940 - metric: 0.250 - time: 13.5 (0.4) 
best
Epoch 3/125
train loss: 0.2237 - metric: 0.195 - time: 19.8 (6.3) 
valid loss: 0.1440 - metric: 0.144 - time: 19.8 (0.3) 
best
Epoch 4/125
train loss: 0.1534 - metric: 0.166 - time: 26.2 (6.4) 
valid loss: 0.1101 - metric: 0.150 - time: 26.2 (0.3) 
Epoch 5/125
train loss: 0.0982 - metric: 0.127 - time: 32.9 (6.7) 
valid loss: 0.0779 - metric: 0.128 - time: 32.9 (0.4) 
best
Epoch 6/125
train loss: 0.0776 - metric: 0.112 - time: 39.5 (6.6) 
valid loss: 0.0753 - metric: 0.127 - time: 39.5 (0.3) 
best
Epoch 7/125
train loss: 0.0730 - metric: 0.112 - time: 45.9 (6.4) 
valid loss: 0.0539 - metric: 0.088 - time: 45.9 (0.3) 
best
Epoch 8/125
train loss: 0.0641 - metric: 0.101 - time: 52.2 (6.3) 
valid loss: 0.0491 - metric: 0.086 - time: 52.2 (0.3) 
best
Epoch 9/125
train loss: 0.0560 - metric: 0.094 - time: 58.6 (6.4) 
valid loss: 0.0451 - metric: 0.081 - time: 58.6 (0.4) 
best
Epoch 10/125
train loss: 0.0513 - metric: 0.089 - time: 64.9 (6.3) 
valid loss: 0.0457 - metric: 0.081 - time: 64.9 (0.4) 
best
Epoch 11/125
train loss: 0.0496 - metric: 0.087 - time: 71.0 (6.1) 
valid loss: 0.0433 - metric: 0.082 - time: 71.0 (0.4) 
Epoch 12/125
train loss: 0.0491 - metric: 0.088 - time: 77.3 (6.3) 
valid loss: 0.0442 - metric: 0.078 - time: 77.3 (0.3) 
best
Epoch 13/125
train loss: 0.0493 - metric: 0.087 - time: 83.5 (6.2) 
valid loss: 0.0442 - metric: 0.085 - time: 83.5 (0.3) 
Epoch 14/125
train loss: 0.0459 - metric: 0.084 - time: 89.7 (6.2) 
valid loss: 0.0404 - metric: 0.076 - time: 89.7 (0.3) 
best
Epoch 15/125
train loss: 0.0440 - metric: 0.082 - time: 96.0 (6.3) 
valid loss: 0.0386 - metric: 0.076 - time: 96.0 (0.3) 
best
Epoch 16/125
train loss: 0.0432 - metric: 0.081 - time: 102.3 (6.3) 
valid loss: 0.0426 - metric: 0.076 - time: 102.3 (0.3) 
Epoch 17/125
train loss: 0.0427 - metric: 0.081 - time: 108.5 (6.2) 
valid loss: 0.0375 - metric: 0.074 - time: 108.5 (0.4) 
best
Epoch 18/125
train loss: 0.0416 - metric: 0.080 - time: 114.5 (6.1) 
valid loss: 0.0408 - metric: 0.073 - time: 114.5 (0.4) 
best
Epoch 19/125
train loss: 0.0417 - metric: 0.080 - time: 120.7 (6.1) 
valid loss: 0.0406 - metric: 0.082 - time: 120.7 (0.3) 
Epoch 20/125
train loss: 0.0408 - metric: 0.079 - time: 126.9 (6.2) 
valid loss: 0.0371 - metric: 0.075 - time: 126.9 (0.3) 
Epoch 21/125
train loss: 0.0396 - metric: 0.078 - time: 133.1 (6.2) 
valid loss: 0.0384 - metric: 0.072 - time: 133.1 (0.4) 
best
Epoch 22/125
train loss: 0.0391 - metric: 0.077 - time: 139.2 (6.1) 
valid loss: 0.0376 - metric: 0.076 - time: 139.2 (0.3) 
Epoch 23/125
train loss: 0.0390 - metric: 0.078 - time: 145.6 (6.4) 
valid loss: 0.0360 - metric: 0.075 - time: 145.6 (0.3) 
Epoch 24/125
train loss: 0.0383 - metric: 0.076 - time: 152.7 (7.1) 
valid loss: 0.0350 - metric: 0.073 - time: 152.7 (0.3) 
Epoch 25/125
train loss: 0.0373 - metric: 0.075 - time: 159.0 (6.3) 
valid loss: 0.0348 - metric: 0.070 - time: 159.0 (0.3) 
best
Epoch 26/125
train loss: 0.0368 - metric: 0.075 - time: 165.4 (6.4) 
valid loss: 0.0341 - metric: 0.070 - time: 165.4 (0.3) 
Epoch 27/125
train loss: 0.0362 - metric: 0.075 - time: 172.4 (7.0) 
valid loss: 0.0332 - metric: 0.067 - time: 172.4 (0.3) 
best
Epoch 28/125
train loss: 0.0366 - metric: 0.075 - time: 178.6 (6.2) 
valid loss: 0.0381 - metric: 0.079 - time: 178.6 (0.4) 
Epoch 29/125
train loss: 0.0369 - metric: 0.075 - time: 184.7 (6.1) 
valid loss: 0.0339 - metric: 0.069 - time: 184.7 (0.3) 
Epoch 30/125
train loss: 0.0350 - metric: 0.074 - time: 190.9 (6.3) 
valid loss: 0.0341 - metric: 0.069 - time: 190.9 (0.3) 
Epoch 31/125
train loss: 0.0340 - metric: 0.072 - time: 197.1 (6.2) 
valid loss: 0.0329 - metric: 0.068 - time: 197.1 (0.3) 
Epoch 32/125
train loss: 0.0339 - metric: 0.072 - time: 203.3 (6.2) 
valid loss: 0.0352 - metric: 0.070 - time: 203.3 (0.3) 
Epoch 33/125
train loss: 0.0343 - metric: 0.073 - time: 209.5 (6.2) 
valid loss: 0.0337 - metric: 0.068 - time: 209.5 (0.3) 
Epoch 34/125
train loss: 0.0341 - metric: 0.072 - time: 215.8 (6.4) 
valid loss: 0.0324 - metric: 0.066 - time: 215.8 (0.3) 
best
Epoch 35/125
train loss: 0.0332 - metric: 0.072 - time: 221.9 (6.1) 
valid loss: 0.0321 - metric: 0.067 - time: 221.9 (0.4) 
Epoch 36/125
"""
