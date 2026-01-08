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
    print(cfg)
    trainer = Trainer(cfg)
    trainer.load_model('/Users/azatgalautdinov/PycharmProjects/price_prediction/interpretation/transformer.pt')
    print(trainer.model.embed.weight.dtype)

    def kv_weights(block):
        k = block.attn.k_compressor.weight.abs().sum(dim=0)
        v = block.attn.v_compressor.weight.abs().sum(dim=0)
        return k + v

    w = [kv_weights(block) for block in trainer.model.blocks]
    ws = w[0]
    for i in range(1, len(w)):
        ws = ws + w[i]
    ids = torch.argsort(ws).numpy()[::-1]
    print(ws.numpy()[ids])
    print(np.array(cfg.data_cfg.features)[ids])


    # num_bins = np.cumsum(cfg.data_cfg.data_transformer.num_bins)
    # # print(num_bins)
    # # print(trainer.models.embed.embed.weight)
    # embed_weight = trainer.models.embed.embed.weight[num_bins[0]: num_bins[1]]
    # # for ew in embed_weight:
    # cs = cosine_similarity(embed_weight, embed_weight)
    # # sns.heatmap(cs) # , cmap="viridis")
    # # # plt.title("Простой Heatmap")
    # # plt.show()
    # plt.imshow(cs, cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    # plt.title("Heatmap с matplotlib")
    # plt.show()


if __name__ == '__main__':
    main()