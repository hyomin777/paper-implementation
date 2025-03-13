import torch.nn as nn
import torch.nn.functional as F


class GoogLeNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, aux1=None, aux2=None):
        main_loss = F.cross_entropy(x, y)

        aux = False        
        aux_loss = 0.0
        if aux1 is not None:
            aux_loss += F.cross_entropy(aux1, y)
            aux = True
        if aux2 is not None:
            aux_loss += F.cross_entropy(aux2, y)
            aux = True
        
        if aux:
            loss = main_loss + (0.3 * aux_loss)
        else:
            loss = main_loss
        
        return loss