import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from accelerate import Accelerator
from data.apartment_dataset import ApartmentDataset
import models.transformers as models
from utils.logger import Logger
from utils.utils import set_seed, get_scheduler, mape, accuracy, get_param_groups


class Trainer:
    def __init__(self, cfg):
        set_seed(cfg.seed)

        self.logger = Logger(cfg)
        self.cfg = cfg

        self.best_epoch = -1
        self.best_loss = float('inf')
        self.time_training = 0

        if cfg.task == 'pretrain':
            self.best_metric = 0
        elif cfg.task == 'train':
            self.best_metric = 1e8
        else:
            raise NotImplementedError()

        self._prepare_data(cfg.data_cfg)
        self._prepare_model(cfg.model_cfg)

        self.logger.print('Training on ' + str(self.accelerator.device))

    def _prepare_data(self, data_cfg):
        self.data_transformer = data_cfg.data_transformer

        self.train_data = ApartmentDataset("train", **data_cfg)
        self.valid_data = ApartmentDataset('valid', **data_cfg)

        kwargs = {'batch_size': self.cfg.batch_size}
        if torch.cuda.is_available():
            kwargs['num_workers'] = 2
            kwargs['pin_memory'] = True
        self.train_dataloader = DataLoader(self.train_data, shuffle=True, **kwargs)
        self.val_dataloader = DataLoader(self.valid_data, shuffle=False, **kwargs)

    def _prepare_model(self, model_cfg):
        self.model = getattr(models, self.cfg.model)(**model_cfg)
        self.criterion = getattr(nn, self.cfg.loss)(**self.cfg.loss_args)
        self.optimizer = getattr(torch.optim, self.cfg.optim)(
            get_param_groups(self.model,
                             self.cfg.lr,
                             self.cfg.weight_decay,
                             self.cfg.get('lr_decay_by_block')),
            **self.cfg.optim_args
        )
        self.scheduler = get_scheduler(self.optimizer, len(self.train_dataloader) * self.cfg.num_epoch,
                                       self.cfg.lr_decay, self.cfg.lr, self.cfg.lr_decay_factor,
                                       self.cfg.wu_ratio, self.cfg.decay_ratio)
        self.accelerator = Accelerator(**self.cfg.accelerator_args)
        (
            self.model, self.optimizer, self.train_dataloader,
            self.val_dataloader, self.scheduler
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader,
            self.val_dataloader, self.scheduler
        )

        pretrained_path = self.cfg.get('load_pretrained', False)
        if pretrained_path:
            self.load_model(pretrained_path, strict=False)
            self.model.zero_compressors_()
            self.logger.print('load_pretrained')

        checkpoint_path = self.cfg.get('load_checkpoint', False)
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            self.logger.print('load_checkpoint')

    def metric(self, pred, label):
        pred = pred.cpu()
        label = label.cpu()
        if self.cfg.task == 'pretrain':
            return accuracy(pred, label)
        elif self.cfg.task == 'train':
            pred = torch.as_tensor(
                self.data_transformer.inverse_transform(pred, target='num')
            )
            return mape(pred, label)
        else:
            raise NotImplementedError()

    def save_model(self, save_path=None, **kwargs):
        if save_path is None:
            save_path = os.path.join(self.cfg.exp_dir, f"{self.cfg.model}.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path, **kwargs)

    def load_model(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = os.path.join(self.cfg.exp_dir, f"{self.cfg.model}.pt")
        self.model.load_state_dict(torch.load(load_path), **kwargs)

    def save_checkpoint(self, save_path=None, **kwargs):
        if save_path is None:
            save_path = os.path.join(self.cfg.exp_dir, "checkpoint")
        self.accelerator.save_state(save_path, **kwargs)

    def load_checkpoint(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = os.path.join(self.cfg.exp_dir, "checkpoint")
        self.accelerator.load_state(load_path, **kwargs)

    def make_step(self, batch, update_model=True):
        with self.accelerator.autocast():
            if self.cfg.task == 'pretrain':
                pred, mask = self.model(batch['features'], self.cfg.mask_ratio)
                batch['target'] = batch['target'][mask]
                batch['label'] = batch['label'][mask]
            elif self.cfg.task == 'train':
                pred = self.model(batch['features'])
            else:
                raise NotImplementedError()
            # print(pred.shape)
            # print(batch['target'].shape)
            loss = self.criterion(pred, batch['target'])

        if update_model:
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        return loss.item(), pred.detach()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_metric = 0
        total_samples = 0

        t = time.time()
        for i, batch in enumerate(self.train_dataloader):
            loss, pred = self.make_step(batch)
            batch_len = len(pred)
            total_samples += batch_len
            total_loss += loss * batch_len
            total_metric += self.metric(pred, batch['label']) * batch_len

        t = time.time() - t
        total_loss /= total_samples
        total_metric /= total_samples
        self.time_training += t

        return {
            'loss': total_loss,
            'metric': total_metric,
            'time': t
        }

    @torch.no_grad()
    def evaluate(self,):
        self.model.eval()
        total_loss = 0
        total_metric = 0
        total_samples = 0

        t = time.time()
        for i, batch in enumerate(self.val_dataloader):
            loss, pred = self.make_step(batch, False)
            batch_len = len(pred)
            total_samples += batch_len
            total_loss += loss * batch_len
            total_metric += self.metric(pred, batch['label']) * batch_len

        t = time.time() - t
        total_loss /= total_samples
        total_metric /= total_samples
        return {
            'loss': total_loss,
            'metric': total_metric,
            'time': t
        }

    def fit(self):
        for epoch in range(1, self.cfg.num_epoch + 1):
            train = self.train_epoch()
            valid = self.evaluate()

            self.logger.log_metrics(epoch, train, valid)
            if (
                    (self.cfg.task == 'train' and valid['metric'] < self.best_metric) or
                    (self.cfg.task == 'pretrain' and valid['loss'] < self.best_loss)
            ):
                self.logger.print('Best')
                self.save_model()
                self.best_metric = valid['metric']
                self.best_loss = valid['loss']
                self.best_epoch = epoch
            else:
                self.logger.print(
                    f"Best | epoch: {self.best_epoch} | metric: "
                    f"{self.best_metric:.5f} | loss: {self.best_loss:.5f}"
                )
            self.save_checkpoint()
            self.logger.save_plot('loss')
            self.logger.save_plot('metric')
            # print(self.optimizer.param_groups[0]['lr'])

    def overfitting_on_batch(self, max_step=1000):
        batch = next(iter(self.train_dataloader))
        for step in range(max_step):
            loss, output = self.make_step(batch, update_model=True)
            if step % 100 == 0:
                self.logger.print(f'[{step}]: loss - {loss:.4f}')


if __name__ == "__main__":
    from configs.train_cfg import cfg
    # from configs.pretrain_cfg import cfg

    trainer = Trainer(cfg)

    # trainer.overfitting_on_batch()
    trainer.fit()
