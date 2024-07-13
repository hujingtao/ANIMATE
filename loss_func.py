import torch.nn as nn
from functools import partial
import torch.nn.functional as F

def sce_loss(x, y, alpha=2):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    # loss = loss.mean()
    return loss

def setup_loss_fn(loss_fn, alpha_l = 2):
    if loss_fn == "mse":
        # print(f"=== Use mse_loss ===")
        criterion = nn.MSELoss(reduction='none')
    elif loss_fn == "sce":
        # print(f"=== Use sce_loss and alpha_l={alpha_l} ===")
        criterion = partial(sce_loss, alpha=alpha_l)
    else:
        raise NotImplementedError
    return criterion


