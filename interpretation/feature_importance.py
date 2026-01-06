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
    trainer = Trainer(cfg)
    trainer.load_model()
    print(trainer.model.embed.embed.weight.dtype)

    def kv_weights(block):
        k = block.attn.k_compressor.weight.abs().sum(dim=0)
        v = block.attn.v_compressor.weight.abs().sum(dim=0)
        return k + v

    w = [kv_weights(block) for block in trainer.model.blocks]
    ids = w[0]
    for i in range(1, len(w)):
        ids = ids + w[i]
    # print(ids)
    ids = torch.argsort(ids).numpy()[::-1]
    print(np.array(cfg.data_cfg.features)[ids])


    # num_bins = np.cumsum(cfg.data_cfg.data_transformer.num_bins)
    # # print(num_bins)
    # # print(trainer.modules.embed.embed.weight)
    # embed_weight = trainer.modules.embed.embed.weight[num_bins[0]: num_bins[1]]
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