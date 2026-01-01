import os

import dill
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.apartment_dataset import ApartmentDataset
# from models.models import PricePredictor
from models.transformer import Transformer
# from utils.metrics import accuracy, balanced_accuracy
from utils.utils import set_seed


class Trainer:
    def __init__(self, cfg):
        set_seed(cfg.seed)
        self.cfg = cfg
        self._prepare_data(cfg.data_cfg)
        self._prepare_model(cfg.model_cfg)

    def _prepare_data(self, data_cfg):
        path = data_cfg.path
        with open(os.path.join(path, "data_transformer.pkl"), 'rb') as f:
            self.data_transformer = dill.load(f)
        with open(os.path.join(path, "target_transformer.pkl"), 'rb') as f:
            self.target_transformer = dill.load(f)
        self.train_data = ApartmentDataset(path, "train", self.data_transformer, self.target_transformer)
        self.valid_data = ApartmentDataset(path, 'valid', self.data_transformer, self.target_transformer)
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.cfg.batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(self.valid_data, batch_size=self.cfg.batch_size, shuffle=False)

    def _mae(self, output, target):
        output = self.target_transformer.inverse_transform(output)
        target = self.target_transformer.inverse_transform(target)

    def _prepare_model(self, model_cfg):
        self.model = torch.compile(Transformer(**model_cfg))
        self.criterion = nn.MSELoss()
        self.optimizer = self.model.configure_optimizers(self.cfg.init_cfg)

    def save_model(self):
        save_path = os.path.join(self.cfg.exp_dir, "transformer.pt")
        torch.save(self.model.state_dict(), save_path)

    def load_model(self):
        load_path = os.path.join(self.cfg.exp_dir, "transformer.pt")
        self.model.load_state_dict(torch.load(load_path))

    def make_step(self, batch, update_model=True):
        outputs = self.model(batch['features'], batch['mask'])
        loss = self.criterion(outputs, batch['target'])

        if update_model:
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item(), outputs

    def train_epoch(self):
        self.model.train()
        for i, batch in enumerate(self.train_dataloader):
            loss, output = self.make_step(batch)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        for i, batch in enumerate(self.train_dataloader):
            loss, output = self.make_step(batch, False)

    def fit(self, *args, **kwargs):
        for epoch in range(self.cfg.num_epochs):  # self.cfg.num_epochs):
            print(f"Epoch {epoch + 1}/{self.cfg.num_epochs}")  # self.cfg.num_epochs}")
            self.train_epoch()
            self.evaluate()

    def overfitting_on_batch(self, max_step=100):
        batch = next(iter(self.train_dataloader))
        for step in range(max_step):
            loss, output = self.make_step(batch, update_model=True)
            if step % 10 == 0:
                acc = accuracy(output, batch['label'])
                print('[{:d}]: loss - {:.4f}, {:.4f}'.format(step, loss, acc))


if __name__ == "__main__":
    from configs.train_cfg import cfg

    trainer = Trainer(cfg)

    trainer.fit()

    trainer.evaluate()