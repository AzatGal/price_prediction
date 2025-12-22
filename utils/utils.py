import torch
import torch.nn as nn


# def init_module(module: nn.Module,
#                 init: nn.Module,
#                 init_args: dict,
#                 out_init_args: dict) -> None:
#     for pn, p in module.named_parameters():
#         if 'out' in pn:
#             init(p, **out_init_args)
#         else:
#             init(p, **init_args)