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


def get_scheduler(optimizer, num_steps, decay, lr, lr_decay_factor):
    wu_steps = int(0.05 * num_steps)
    num_steps -= wu_steps
    decay_steps = int(0.5 * num_steps)
    num_steps -= decay_steps
    schedulers = [
        torch.optim.lr_scheduler.LinearLR(optimizer,
                                          total_iters=wu_steps,
                                          start_factor=lr * lr_decay_factor,
                                          end_factor=1.0)
    ]
    if decay == 'linear':
        schedulers.append(
            torch.optim.lr_scheduler.LinearLR(optimizer,
                                              total_iters=decay_steps,
                                              start_factor=1.0,
                                              end_factor=lr_decay_factor)
        )
    elif decay == 'cosine':
        schedulers.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=decay_steps,
                                                       eta_min=lr * lr_decay_factor)
        )
    else:
        raise NotImplementedError()
    schedulers.append(
        torch.optim.lr_scheduler.LinearLR(optimizer,
                                          total_iters=num_steps,
                                          start_factor=lr_decay_factor,
                                          end_factor=lr_decay_factor / 10)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=[wu_steps, wu_steps + decay_steps],
    )


def accuracy(pred, label):
    pred = pred.argmax(1)
    return torch.sum(pred == label).item()


def mape(pred, label, epsilon=1e-8):
    return torch.abs(
        (label - pred) / (label + epsilon)
    ).sum().item()


if __name__ == '__main__':
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=10)
    for epoch in range(12):
        print(epoch, opt.param_groups[0]['lr'])
        scheduler.step()

