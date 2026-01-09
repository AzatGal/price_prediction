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
train loss: 0.7999 - metric: 0.487 - time: 7.3 (7.3) 
valid loss: 0.8201 - metric: 0.496 - time: 7.3 (0.3) 
best
Epoch 2/125
train loss: 0.5164 - metric: 0.296 - time: 14.6 (7.2) 
valid loss: 0.3990 - metric: 0.188 - time: 14.6 (0.4) 
best
Epoch 3/125
train loss: 0.2732 - metric: 0.147 - time: 22.3 (7.7) 
valid loss: 0.3086 - metric: 0.105 - time: 22.3 (0.3) 
best
Epoch 4/125
train loss: 0.2629 - metric: 0.145 - time: 29.9 (7.6) 
valid loss: 0.2703 - metric: 0.094 - time: 29.9 (0.3) 
best
Epoch 5/125
train loss: 0.2206 - metric: 0.119 - time: 39.3 (9.4) 
valid loss: 0.2952 - metric: 0.104 - time: 39.3 (0.3) 
Epoch 6/125
train loss: 0.2076 - metric: 0.112 - time: 47.0 (7.7) 
valid loss: 0.2731 - metric: 0.087 - time: 47.0 (0.3) 
best
Epoch 7/125
train loss: 0.1892 - metric: 0.103 - time: 54.4 (7.5) 
valid loss: 0.2696 - metric: 0.092 - time: 54.4 (0.4) 
Epoch 8/125
train loss: 0.1778 - metric: 0.096 - time: 63.8 (9.3) 
valid loss: 0.2645 - metric: 0.082 - time: 63.8 (0.5) 
best
Epoch 9/125
train loss: 0.1694 - metric: 0.092 - time: 71.4 (7.7) 
valid loss: 0.2501 - metric: 0.086 - time: 71.4 (0.4) 
Epoch 10/125
train loss: 0.1632 - metric: 0.089 - time: 80.0 (8.6) 
valid loss: 0.2928 - metric: 0.093 - time: 80.0 (0.3) 
Epoch 11/125
train loss: 0.1500 - metric: 0.081 - time: 89.2 (9.2) 
valid loss: 0.2132 - metric: 0.078 - time: 89.2 (0.4) 
best
Epoch 12/125
"""
