"""@AikenH 2021 loss selection design part
LOSS FUNCTION SELECTION
"""
# import torch loss function
import torch.nn as nn 
# import loss fuction u write in this dir

def L_select(loss_t,*args,**kwargs):
    if loss_t == 'MSE':
        return nn.MSELoss()
    elif loss_t == 'L1':
        return nn.L1Loss()
    elif loss_t == 'Huber':
        return nn.SmoothL1Loss()
    elif loss_t == 'BCE':
        return nn.BCELoss()
    elif loss_t == 'CE':
        return nn.CrossEntropyLoss()
    elif loss_t == 'NLL':
        return nn.NLLLoss()
    else:
        raise NotImplementedError('NOT IMPLEMENTED, add it in L_select')
        