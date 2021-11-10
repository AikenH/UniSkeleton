"""
@AIkenH 2021 ACTIVATIONs
@Desc: Write those Activation here and register it in the Interfrate Function
"""
import torch
from torch import nn
from torch.nn import functional as F 

class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()
        # TEST
        # print("Mish activation loader")
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

# Swish 的beta是不是一个可训练参数？
# 在这里只写了默认参数是1的情况
class Swish(nn.Module):
    def __init__(self,beta=1):
        super(Swish,self).__init__()
        self.beta = beta

    def forward(self,x):
        # x  *= torch.sigmoid(self.beta * x)
        return x * torch.sigmoid(x)

def activation_loader(act_name):
    if act_name is None:
        return None
    elif act_name == 'Mish':
        return Mish()
    elif act_name == 'Swish':
        return Swish()
    elif act_name.lower() == 'relu':
        return nn.ReLU(inplace=True)
    elif act_name.lower() == 'gelu':
        return F.gelu
    else:
        raise NotImplementedError