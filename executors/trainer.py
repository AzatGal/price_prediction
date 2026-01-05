import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from accelerate import Accelerator
from data.apartment_dataset import ApartmentDataset
from models.transformer import Transformer
from utils.utils import set_seed, get_scheduler, mape_loss


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
        self.best_loss = float('inf')
        self.best_metric = float('inf')
        self.best_epoch = 0

    def _prepare_data(self, data_cfg):
        self.data_transformer = data_cfg.data_transformer
        self.train_data = ApartmentDataset("train", data_cfg.path, self.data_transformer,
                                           data_cfg.target, data_cfg.smooth)
        self.val_data = ApartmentDataset('valid', data_cfg.path, self.data_transformer,
                                         data_cfg.target, data_cfg.smooth)
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.cfg.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_data, batch_size=self.cfg.batch_size, shuffle=False)

    def _prepare_model(self, model_cfg):
        self.model = Transformer(**model_cfg)
        self.criterion = getattr(nn, self.cfg.loss)(**self.cfg.loss_args)
        self.optimizer = self.model.configure_optimizer(self.cfg.lr, self.cfg.weight_decay)
        self.scheduler = get_scheduler(self.optimizer, len(self.train_dataloader) * self.cfg.num_epoch)

    def metric(self, pred, label):
        if pred.shape[-1] > 1:
            pred = pred.argmax(-1)
        pred = torch.as_tensor(
            self.data_transformer.inverse_transform(pred, target=True)
        )
        return mape_loss(pred, label)

    def save_model(self):
        save_path = os.path.join(self.cfg.exp_dir, "transformer.pt")
        torch.save(self.model.state_dict(), save_path)

    def load_model(self):
        load_path = os.path.join(self.cfg.exp_dir, "transformer.pt")
        self.model.load_state_dict(torch.load(load_path))

    def make_step(self, batch, update_model=True):
        with self.accelerator.autocast():
            pred = self.model(batch['features'], batch['mask'])
            loss = self.criterion(pred, batch['target'])

        if update_model:
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        return loss.item(), pred.detach()

    def _print(self, s, l, m, t):
        print(f'{s} loss: {l:.4f} - metric: {m:.3f} - time: {t:.1f} ')

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_metric = 0
        t = time.time()
        for i, batch in enumerate(self.train_dataloader):
            loss, pred = self.make_step(batch)
            total_loss += loss
            total_metric += self.metric(pred, batch['label'])

        total_loss /= len(self.train_dataloader)
        total_metric /= len(self.train_dataloader)
        self._print('train', total_loss, total_metric, time.time() - t)

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_metric = 0
        t = time.time()
        for i, batch in enumerate(self.val_dataloader):
            loss, pred = self.make_step(batch, False)
            total_loss += loss
            total_metric += self.metric(pred, batch['label'])

        total_loss /= len(self.val_dataloader)
        total_metric /= len(self.val_dataloader)
        self._print('valid', total_loss, total_metric, time.time() - t)
        if total_metric < self.best_metric:
            print('best')
            self.save_model()
            self.best_metric = total_metric
            self.best_loss = total_loss
            self.best_epoch = epoch

    def fit(self):
        for epoch in range(self.cfg.num_epoch):
            print(f"Epoch {epoch + 1}/{self.cfg.num_epoch}")
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

    trainer = Trainer(cfg)
    # trainer.overfitting_on_batch()
    trainer.fit()
    # trainer.evaluate()
