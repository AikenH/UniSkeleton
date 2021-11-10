import torch
import numpy as np 
import torch.nn.functional as F 

# maybe used in distill or incremental learning
def kl_loss(pred,real,**kwargs):
    """ caculate the kl-divergence between label and pred

    Args:
        pred ([type]): [description]
        real ([type]): [description]
    """
    pred_copy = F.log_softmax(pred,dim =1 )
    kl_loss = F.kl_div(pred_copy,real,reduction='batchmean')
    return kl_loss


