import torch
from torch import nn

import numpy as np 
import torch.nn.functional as F 


def NIG_NLL(y,gamma,v,alpha,beta,reduce=True):
    twoBlambda = 2.*beta*(1.+v)

    nll =  0.5*torch.log(np.pi/v)  \
        - alpha*torch.log(twoBlambda)  \
        + (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)
    
    return torch.mean(nll) if reduce else nll 

def NIG_Reg(y,gamma,v,alpha,beta,omega,reduce=True,k1=False):
    error = torch.abs(y-gamma)

    if k1:
        kl = KL_NIG(gamma,v,alpha,beta,gamma,omega,1+omega,beta)
        reg = error * kl
    else:
        evi = 2*v + alpha
        reg = error*evi
    
    return torch.mean(reg) if reduce else reg

def KL_NIG(mu1,v1,a1,b1,mu2,v2,a2,b2):
    KL = 0.5 * (a1-1)/b1 * (v2*torch.square(mu2-mu1)) \
        + 0.5* v2/v1 \
        - 0.5* torch.log(torch.abs(v2)/torch.abs(v1)) \
        - 0.5+ a2*torch.log(b1/b2) \
        - (torch.lgamma(a1)-torch.lgamma(a2)) \
        + (a1-a2)* torch.digamma(a1) \
        - (b1-b2)*a1/b1
    return KL

def EvidentialRegression(y_true, evidential_output, coeff=1.0):
    gamma, v, alpha, beta = torch.split(evidential_output, 4, axis=-1)
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    return loss_nll + coeff * loss_reg
