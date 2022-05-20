"""
we use the arcface loss to replace ce in some situations.
bcus the normalized operation had been finished in our classifier,
we only code the calculation of the final loss part.

@reference: 
    https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch

use the calculation here.
"""
import torch
import torch.nn as nn

class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type='arcface', eps=1e-7, s=None, m=None):
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        
        # got the default parameters
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        elif loss_type == 'sphereface':
            self.s = 64.0 if not s else s 
            self.m = 1.35 if not m else m
        elif loss_type == 'cosface':
            self.s = 30.0 if not s else s 
            self.m = 0.4 if not m else m

        self.loss_type = loss_type
        self.eps = eps
    
    def forward(self, pred, labels):
        """intergrate those three loss"""
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(pred.transpose(0,1)[labels]) - self.m)
        elif self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(pred.transpose(0,1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        elif self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(pred.transpose(0,1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((pred[i,:y], pred[i,y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl),dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
