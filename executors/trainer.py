import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import rtdl_num_embeddings

from torch.utils.data import DataLoader
from accelerate import Accelerator
from dataset.apartment_dataset.apartment_dataset import ApartmentDataset
from models.transformers import TransformerEnsemble
from utils.logger import Logger
from utils.utils import set_seed, get_scheduler, mape, get_param_groups


class Trainer:
    def __init__(self, cfg, log=True):
        set_seed(cfg.seed)

        self.accelerator = Accelerator(**cfg.accelerator_args)
        if log:
            self.logger = Logger(cfg)
            self.logger.print('Training on ' + str(self.accelerator.device))
        self.cfg = cfg

        self.best_epoch = -1
        self.best_loss = float('inf')
        self.time_training = 0
        self.start_epoch = 1

        self.best_metric = float('inf')

        self._prepare_data(cfg.data_cfg)
        self._prepare_model(cfg.model_cfg)

    def _prepare_data(self, data_cfg):
        self.train_dataset = ApartmentDataset(
            data_cfg.processors.num.fit_transform(data_cfg.raw_data.train.num),
            data_cfg.processors.cat.fit_transform(data_cfg.raw_data.train.cat),
            data_cfg.processors.target.fit_transform(data_cfg.raw_data.train.label.to_numpy()),
            data_cfg.raw_data.train.label.to_numpy(),
        )
        self.val_dataset = ApartmentDataset(
            data_cfg.processors.num.transform(data_cfg.raw_data.val.num),
            data_cfg.processors.cat.transform(data_cfg.raw_data.val.cat),
            data_cfg.processors.target.transform(data_cfg.raw_data.val.label.to_numpy()),
            data_cfg.raw_data.val.label.to_numpy(),
        )

        self.target_processor = data_cfg.processors.target
        # print(data_cfg.processors.num.steps[1][1].n_bins_.tolist())

        self.cfg.model_cfg.n_embed_num = (
            data_cfg.processors.num.steps[1][1].n_bins_.tolist()
            if hasattr(data_cfg.processors.num.steps[1][1], 'n_bins_')
            else data_cfg.raw_data.train.num.shape[1]
        )
        self.cfg.model_cfg.n_embed_cat = [
            len(cat) + 1 for cat in data_cfg.processors.cat.steps[1][1].categories_
        ]
        # print(self.cfg.model_cfg.n_embed_num)
        # print(self.cfg.model_cfg.n_embed_cat)

        kwargs = {'batch_size': self.cfg.batch_size}
        if torch.cuda.is_available():
            kwargs['num_workers'] = 2
            kwargs['pin_memory'] = True
        self.train_dataloader = DataLoader(self.train_dataset, shuffle=True, **kwargs)
        self.val_dataloader = DataLoader(self.val_dataset, shuffle=False, **kwargs)

    def _prepare_model(self, model_cfg):
        self.model = TransformerEnsemble(**model_cfg)

        # self.model.embed.num_embed = nn.ModuleList([
        #     rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        #         rtdl_num_embeddings.compute_bins(
        #             torch.cat([x['x_num'].unsqueeze(0) for x in self.train_dataset]),
        #             # n_bins=128 # 48
        #         ),
        #         d_embedding=model_cfg.embed_dim,
        #         activation=True,  # False,
        #         version='B',
        #     )
        #     for _ in range(1)  # model_cfg.k)
        # ])

        self.criterion = getattr(nn, self.cfg.loss)(**self.cfg.loss_args)
        self.optimizer = getattr(torch.optim, self.cfg.optim)(
            get_param_groups(self.model,
                             self.cfg.lr,
                             self.cfg.weight_decay),
            **self.cfg.optim_args
        )
        self.scheduler = get_scheduler(self.optimizer, len(self.train_dataloader) * self.cfg.num_epoch,
                                       self.cfg.lr_decay, self.cfg.lr, self.cfg.lr_decay_factor,
                                       self.cfg.wu_ratio, self.cfg.decay_ratio)
        (
            self.model, self.optimizer, self.train_dataloader,
            self.val_dataloader, self.scheduler
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader,
            self.val_dataloader, self.scheduler
        )

        # pretrained_path = self.cfg.get('load_pretrained', False)
        # if pretrained_path:
        #     self.load_model(pretrained_path, strict=False)
        #     self.logger.print('load_pretrained')
        #
        # checkpoint_path = self.cfg.get('load_checkpoint', False)
        # if checkpoint_path:
        #     self.load_checkpoint(checkpoint_path)
        #     self.logger.print('load_checkpoint')

    def metric(self, pred, label):
        pred = pred.cpu()
        label = label.cpu()

        pred = torch.as_tensor(self.target_processor.inverse_transform(pred))
        # return mape(pred, label)
        pred = pred.mean(1, keepdim=True)
        return torch.sqrt(F.mse_loss(pred, label)).item()

    def save_model(self, save_path=None, **kwargs):
        if save_path is None:
            save_path = os.path.join(self.cfg.exp_dir, "model.pt")  # {self.cfg.model}
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path, **kwargs)

    def load_model(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = os.path.join(self.cfg.exp_dir, f"{self.cfg.model}.pt")
        self.model.load_state_dict(
            torch.load(load_path,
                       map_location=self.accelerator.device,
                       **kwargs)
        )

    def save_checkpoint(self, epoch, save_path=None, **kwargs):
        if save_path is None:
            save_path = os.path.join(self.cfg.exp_dir, "checkpoint")
        self.accelerator.save_state(save_path, **kwargs)
        with open(os.path.join(save_path, "extra_states.json"), "w") as f:
            json.dump(
                {
                    'epoch': epoch,
                    'best_epoch': self.best_epoch,
                    'best_loss': self.best_loss,
                    'best_metric': self.best_metric,
                    'time_training': self.time_training
                },
                f
            )

    def load_checkpoint(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = os.path.join(self.cfg.exp_dir, "checkpoint")
        self.accelerator.load_state(load_path, **kwargs)

        with open(os.path.join(load_path, "extra_states.json")) as f:
            extra_states = json.load(f)
            self.start_epoch = extra_states['epoch'] + 1
            self.best_epoch = extra_states['best_epoch']
            self.best_loss = extra_states['best_loss']
            self.best_metric = extra_states['best_metric']
            self.time_training = extra_states['time_training']

    def make_step(self, batch, update_model=True):
        with self.accelerator.autocast():
            pred = self.model(batch['x_num'], batch['x_cat']).squeeze(2, 3)
            batch['target'] = batch['target'].repeat(1, pred.size(1))

            loss = self.criterion(pred, batch['target'])

        if update_model:
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        # pred = pred.mean(1, keepdim=True)
        return loss.item(), pred.detach()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_metric = 0
        total_samples = 0

        t = time.time()
        for batch in self.train_dataloader:
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

    @torch.inference_mode()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_metric = 0
        total_samples = 0

        t = time.time()
        for batch in self.val_dataloader:
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
        for epoch in range(self.start_epoch, self.cfg.num_epoch + 1):
            train = self.train_epoch()
            val = self.evaluate()
            self.logger.log_metrics(epoch, train, val)
            if val['metric'] < self.best_metric:
                self.logger.print('Best')
                self.save_model()
                self.best_metric = val['metric']
                self.best_loss = val['loss']
                self.best_epoch = epoch
            else:
                self.logger.print(
                    f"Best | epoch: {self.best_epoch} | metric: "
                    f"{self.best_metric:.5f} | loss: {self.best_loss:.5f}"
                )
            self.save_checkpoint(epoch)
            self.logger.save_plot('loss')
            self.logger.save_plot('metric')

    def overfitting_on_batch(self, max_step=1000):
        batch = next(iter(self.train_dataloader))
        for step in range(max_step):
            loss, output = self.make_step(batch, update_model=True)
            if step % 100 == 0:
                self.logger.print(f'[{step}]: loss - {loss:.4f}')

    # @torch.no_grad()
    # def test(self):
    #     kwargs = {'batch_size': self.cfg.batch_size}
    #     if torch.cuda.is_available():
    #         kwargs['num_workers'] = 2
    #         kwargs['pin_memory'] = True
    #     dataloader = DataLoader(
    #         ApartmentDataset("test", **self.cfg.data_cfg),
    #         shuffle=True, **kwargs
    #     )
    #     dataloader = self.accelerator.prepare(dataloader)
    #
    #     self.model.eval()
    #     total_loss = 0
    #     total_metric = 0
    #     total_samples = 0
    #
    #     t = time.time()
    #     for batch in dataloader:
    #         loss, pred = self.make_step(batch, False)
    #         batch_len = len(pred)
    #         total_samples += batch_len
    #         total_loss += loss * batch_len
    #         total_metric += self.metric(pred, batch['label']) * batch_len
    #
    #     t = time.time() - t
    #     total_loss /= total_samples
    #     total_metric /= total_samples
    #     return {
    #         'loss': total_loss,
    #         'metric': total_metric,
    #         'time': t
    #     }


if __name__ == "__main__":
    from configs.train_cfg import cfg

    # # cfg.accelerator_args['mps'] = True
    # cfg.batch_size = 16
    # cfg.model = 'TablePredictor'
    # path = '/Users/azatgalautdinov/Downloads'
    # # with open(os.path.join(path, 'logs', 'config.json'), 'r') as f:
    # #     model_cfg = json.load(f)['model_cfg']
    # #
    # # model_cfg['add_cls_token'] = model_cfg.pop('add_first_token')
    # # cfg.model_cfg = model_cfg
    # trainer = Trainer(cfg)
    # trainer.load_model(os.path.join(path, 'TablePredictor.pt'))
    #
    # print(trainer.test())

    trainer = Trainer(cfg)
    # trainer.save_model('./test.pt')
    trainer.fit()
