import torch
import torch.nn.functional as F
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

from executors.trainer import Trainer
from configs.train_cfg import cfg


@torch.no_grad()
def main():
    # print(cfg)
    trainer = Trainer(cfg)
    trainer.load_model('/Users/azatgalautdinov/PycharmProjects/price_prediction/runs/train/15-01_22-57/PricePrediction.pt')
    # print(trainer.model.embed.weight.dtype)

    def kv_weights(block):
        kv = [x.weight.abs().sum(dim=0) for x in block.attn.kv_compressors]
        return sum(kv)

    w = [kv_weights(block) for block in trainer.model.blocks]
    ws = w[0]
    for i in range(1, len(w)):
        ws = ws + w[i]
    ids = torch.argsort(ws).numpy()[::-1]
    features = (cfg.data_cfg.data_transformer.num_cols +
                cfg.data_cfg.data_transformer.cat_cols)
    print(ws.numpy()[ids])
    print(np.array(features[0 if cfg.data_cfg.include_target else 1:])[ids])



if __name__ == '__main__':
    main()
