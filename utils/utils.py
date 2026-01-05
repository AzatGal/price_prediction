import random
import numpy as np
import torch
# import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_scheduler(optimizer, num_steps, decay):
    wu_iters = int(0.05 * num_steps)
    d_iters = num_steps - wu_iters
    min_lr = 1e-6
    schedulers = [
        torch.optim.lr_scheduler.LinearLR(optimizer,
                                          total_iters=wu_iters,
                                          start_factor=min_lr,
                                          end_factor=1.0)
    ]
    if decay == 'linear':
        schedulers.append(
            torch.optim.lr_scheduler.LinearLR(optimizer,
                                              total_iters=d_iters,
                                              start_factor=1.0,
                                              end_factor=min_lr)
        )
    elif decay == 'cosine':
        schedulers.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=d_iters,
                                                       eta_min=min_lr)
        )
    else:
        raise NotImplementedError()
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=[wu_iters],
    )


def mape_loss(pred, target, epsilon=1e-8):
    return torch.mean(
        torch.abs(
            (target - pred) / (target + epsilon)
        )
    )


# def smape_loss(pred, target, eps=1e-8):
#     numerator = torch.abs(pred - target)
#     denominator = (pred.abs() + target.abs()) / 2.0 + eps
#     return torch.mean(numerator / denominator)

