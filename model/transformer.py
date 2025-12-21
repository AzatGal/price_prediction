import torch
import torch.nn as nn


class AttnBlock(nn.Module):
    def __init__(self, n_head: int, hidden_dim: int, attn_dropout: float):
