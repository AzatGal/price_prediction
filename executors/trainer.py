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
            pred = self.model(batch['features'], batch.get('mask'))
            if self.cfg.target_type == 'mask':
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
            batch_len = sum(pred.shape[:2])  # * self.num_masks
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
            batch_len = sum(pred.shape[:2])  # * self.num_masks
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

    # with torch.no_grad():
    #     for block in trainer.model.blocks:
    #         block.attn.k_compressor.weight.zero_()
    #         block.attn.v_compressor.weight.zero_()
    # trainer.overfitting_on_batch()
    trainer.fit()

"""
cpu
Epoch 1/100
train loss: 1.2907 - metric: 0.563 - time: 1.9 (1.9) 
valid loss: 0.9904 - metric: 0.499 - time: 1.9 (0.1) 
best
Epoch 2/100
train loss: 0.8653 - metric: 0.434 - time: 3.6 (1.7) 
valid loss: 0.4950 - metric: 0.305 - time: 3.6 (0.1) 
best
Epoch 3/100
train loss: 0.4401 - metric: 0.282 - time: 5.3 (1.7) 
valid loss: 0.3324 - metric: 0.256 - time: 5.3 (0.1) 
best
Epoch 4/100
train loss: 0.3222 - metric: 0.243 - time: 7.1 (1.8) 
valid loss: 0.2102 - metric: 0.211 - time: 7.1 (0.1) 
best
Epoch 5/100
train loss: 0.2231 - metric: 0.191 - time: 9.0 (1.9) 
valid loss: 0.1525 - metric: 0.160 - time: 9.0 (0.1) 
best
Epoch 6/100
train loss: 0.1587 - metric: 0.163 - time: 11.0 (2.0) 
valid loss: 0.1180 - metric: 0.143 - time: 11.0 (0.1) 
best
Epoch 7/100
train loss: 0.1238 - metric: 0.139 - time: 12.8 (1.8) 
valid loss: 0.0947 - metric: 0.120 - time: 12.8 (0.1) 
best
Epoch 8/100
train loss: 0.1016 - metric: 0.125 - time: 14.5 (1.8) 
valid loss: 0.0800 - metric: 0.112 - time: 14.5 (0.1) 
best
Epoch 9/100
train loss: 0.0834 - metric: 0.113 - time: 16.5 (1.9) 
valid loss: 0.0706 - metric: 0.110 - time: 16.5 (0.1) 
best
Epoch 10/100
train loss: 0.0743 - metric: 0.108 - time: 18.2 (1.8) 
valid loss: 0.0584 - metric: 0.094 - time: 18.2 (0.1) 
best
Epoch 11/100
train loss: 0.0645 - metric: 0.099 - time: 20.0 (1.7) 
valid loss: 0.0540 - metric: 0.094 - time: 20.0 (0.1) 
Epoch 12/100
train loss: 0.0599 - metric: 0.095 - time: 21.8 (1.8) 
valid loss: 0.0506 - metric: 0.090 - time: 21.8 (0.1) 
best
Epoch 13/100
train loss: 0.0565 - metric: 0.093 - time: 23.5 (1.7) 
valid loss: 0.0480 - metric: 0.085 - time: 23.5 (0.1) 
best
Epoch 14/100
train loss: 0.0543 - metric: 0.091 - time: 25.2 (1.7) 
valid loss: 0.0460 - metric: 0.084 - time: 25.2 (0.1) 
best
Epoch 15/100
train loss: 0.0524 - metric: 0.089 - time: 27.0 (1.8) 
valid loss: 0.0454 - metric: 0.083 - time: 27.0 (0.1) 
best
Epoch 16/100
train loss: 0.0510 - metric: 0.088 - time: 28.7 (1.7) 
valid loss: 0.0475 - metric: 0.090 - time: 28.7 (0.1) 
Epoch 17/100
train loss: 0.0528 - metric: 0.091 - time: 30.5 (1.8) 
valid loss: 0.0552 - metric: 0.103 - time: 30.5 (0.1) 
Epoch 18/100
train loss: 0.0536 - metric: 0.092 - time: 32.2 (1.7) 
valid loss: 0.0470 - metric: 0.084 - time: 32.2 (0.1) 
Epoch 19/100
train loss: 0.0513 - metric: 0.089 - time: 34.0 (1.8) 
valid loss: 0.0458 - metric: 0.081 - time: 34.0 (0.1) 
best
Epoch 20/100
train loss: 0.0502 - metric: 0.088 - time: 35.8 (1.8) 
valid loss: 0.0445 - metric: 0.085 - time: 35.8 (0.1) 
Epoch 21/100
train loss: 0.0483 - metric: 0.087 - time: 37.7 (1.9) 
valid loss: 0.0447 - metric: 0.085 - time: 37.7 (0.1) 
Epoch 22/100
train loss: 0.0477 - metric: 0.085 - time: 39.5 (1.8) 
valid loss: 0.0434 - metric: 0.083 - time: 39.5 (0.1) 
Epoch 23/100
train loss: 0.0469 - metric: 0.084 - time: 41.3 (1.8) 
valid loss: 0.0421 - metric: 0.079 - time: 41.3 (0.1) 
best
Epoch 24/100
train loss: 0.0457 - metric: 0.084 - time: 43.2 (1.9) 
valid loss: 0.0446 - metric: 0.079 - time: 43.2 (0.1) 
best
Epoch 25/100
train loss: 0.0469 - metric: 0.085 - time: 45.1 (1.9) 
valid loss: 0.0427 - metric: 0.079 - time: 45.1 (0.1) 
Epoch 26/100
train loss: 0.0457 - metric: 0.084 - time: 46.8 (1.8) 
valid loss: 0.0440 - metric: 0.083 - time: 46.8 (0.1) 
Epoch 27/100
train loss: 0.0451 - metric: 0.083 - time: 48.6 (1.8) 
valid loss: 0.0413 - metric: 0.078 - time: 48.6 (0.1) 
best
Epoch 28/100
train loss: 0.0440 - metric: 0.082 - time: 50.4 (1.8) 
valid loss: 0.0408 - metric: 0.076 - time: 50.4 (0.1) 
best
Epoch 29/100
train loss: 0.0429 - metric: 0.080 - time: 52.3 (2.0) 
valid loss: 0.0405 - metric: 0.077 - time: 52.3 (0.1) 
Epoch 30/100
train loss: 0.0426 - metric: 0.080 - time: 54.1 (1.8) 
valid loss: 0.0406 - metric: 0.076 - time: 54.1 (0.1) 
Epoch 31/100
train loss: 0.0421 - metric: 0.080 - time: 56.0 (1.8) 
valid loss: 0.0406 - metric: 0.079 - time: 56.0 (0.1) 
Epoch 32/100
train loss: 0.0419 - metric: 0.080 - time: 57.8 (1.8) 
valid loss: 0.0403 - metric: 0.076 - time: 57.8 (0.1) 
Epoch 33/100
train loss: 0.0426 - metric: 0.081 - time: 59.8 (2.0) 
valid loss: 0.0406 - metric: 0.075 - time: 59.8 (0.1) 
best
Epoch 34/100
train loss: 0.0437 - metric: 0.082 - time: 61.7 (1.8) 
valid loss: 0.0443 - metric: 0.084 - time: 61.7 (0.1) 
Epoch 35/100
train loss: 0.0442 - metric: 0.083 - time: 63.5 (1.8) 
valid loss: 0.0410 - metric: 0.076 - time: 63.5 (0.1) 
Epoch 36/100
train loss: 0.0414 - metric: 0.080 - time: 65.2 (1.8) 
valid loss: 0.0400 - metric: 0.076 - time: 65.2 (0.1) 
Epoch 37/100
train loss: 0.0409 - metric: 0.079 - time: 67.3 (2.0) 
valid loss: 0.0389 - metric: 0.076 - time: 67.3 (0.1) 
Epoch 38/100
train loss: 0.0405 - metric: 0.079 - time: 69.2 (1.9) 
valid loss: 0.0389 - metric: 0.075 - time: 69.2 (0.1) 
best
Epoch 39/100
train loss: 0.0403 - metric: 0.079 - time: 71.1 (1.9) 
valid loss: 0.0385 - metric: 0.074 - time: 71.1 (0.2) 
best
Epoch 40/100
train loss: 0.0395 - metric: 0.078 - time: 72.9 (1.8) 
valid loss: 0.0388 - metric: 0.076 - time: 72.9 (0.1) 
Epoch 41/100
train loss: 0.0394 - metric: 0.078 - time: 74.7 (1.8) 
valid loss: 0.0382 - metric: 0.074 - time: 74.7 (0.1) 
Epoch 42/100
train loss: 0.0389 - metric: 0.077 - time: 76.5 (1.8) 
valid loss: 0.0377 - metric: 0.073 - time: 76.5 (0.1) 
best
Epoch 43/100
train loss: 0.0387 - metric: 0.077 - time: 78.3 (1.8) 
valid loss: 0.0381 - metric: 0.075 - time: 78.3 (0.1) 
Epoch 44/100

"""
