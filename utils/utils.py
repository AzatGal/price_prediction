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


def get_scheduler(optimizer, num_steps, decay, min_lr):
    wu_steps = int(0.05 * num_steps)
    decay_steps = num_steps - wu_steps
    schedulers = [
        torch.optim.lr_scheduler.LinearLR(optimizer,
                                          total_iters=wu_steps,
                                          start_factor=min_lr,
                                          end_factor=1.0)
    ]
    if decay == 'linear':
        schedulers.append(
            torch.optim.lr_scheduler.LinearLR(optimizer,
                                              total_iters=decay_steps,
                                              start_factor=1.0,
                                              end_factor=min_lr)
        )
    elif decay == 'cosine':
        schedulers.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=decay_steps,
                                                       eta_min=min_lr)
        )
    else:
        raise NotImplementedError()
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=[wu_steps],
    )


def accuracy(pred, label):
    pred = pred.argmax(1)
    return torch.sum(pred == label).item()


def mape(pred, label, epsilon=1e-8):
    return torch.abs(
        (label - pred) / (label + epsilon)
    ).sum().item()



# def smape_loss(pred, target, eps=1e-8):
#     numerator = torch.abs(pred - target)
#     denominator = (pred.abs() + target.abs()) / 2.0 + eps
#     return torch.mean(numerator / denominator)

