""" 
@Author: AikenHong
@Purpose: Define some basic custom loss function
"""
import torch
import numpy as np 
import torch.nn.functional as F 

# maybe used in distill or incremental learning
def kl_loss(pred,real,**kwargs):
    """ 
    caculate the kl-divergence between label and pred
    """
    pred_copy = F.log_softmax(pred, dim=1)
    kl_loss = F.kl_div(pred_copy,real,reduction='batchmean')
    return kl_loss

class mixup_criterion():
    def __init__(self, criterion):
        self.criterion = criterion
    
    def __call__(self, pred, y_bf, y_af, lam):
        """
        Mixup criterion for classification, which cannot be used alone.
        We need to use this criterion together with other Loss Function.  

        args:
            criterion: loss function
            pred: prediction
            y_bf: target before shuffled
            y_af: target after shuffled
            lam: mixing parameter
        Return:
            mixup loss
        """
        # notice that if we using mixup, the acc1 cannot be calculate in training 
        return lam * self.criterion(pred, y_bf) + (1 - lam) * self.criterion(pred, y_af)

class EpoesCombiner():
    """
    combine the loss function in the way we want
    """
    def __init__(self, epoches, **kwargs):
        self.epoches = epoches
        self.startegy = kwargs.get('startegy','parabolic')
        self.init_parameters()
    
    def init_parameters(self):
        # why should we do this for 90 and 180?
        self.beta = 0.2
        if self.epoches in [90,180]:
            self.div_epoch = 100 * (self.epoches // 100 +1)
        else:
            self.div_epoch = self.epoches
        return None
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        return None
    
    def get_alpha(self, epoch, **kwargs):
        self.set_epoch(epoch)
        if self.startegy == 'parabolic':
            self.alpha = 1 - ((self.epoch -1 )/ self.div_epoch)**2

        elif self.startegy == 'linear':
            self.alpha = 1 - (self.epoch -1) / self.div_epoch

        elif self.startegy == 'fix':
            self.alpha = 0.5

        elif self.startegy == 'beta':
            self.alpha = np.random.beta(self.beta, self.beta)

        elif self.startegy == 'cosine':
            import math
            self.alpha = math.cos((self.epoch-1) / self.div_epoch * math.pi /2)

        elif self.startegy == 'separate':
            threshold = kwargs.get('threshold', 30)
            self.alpha = 0.5 if threshold < self.epoch else 0

        elif self.startegy == 'abort':
            self.alpha = 0 

        elif self.strategy == 'new_linear':
            self.alpha = 0.4 - 0.4 * (self.epoch-1) / self.div_epoch
            # seems like the best result of now is 0.7
            # we need to reshow this acc.
            # self.alpha = 0.8-0.8*(self.epoch-1) / self.div_epoch
            
        else:
            raise NotImplementedError

        return self.alpha

    def __call__(self, epoch, loss1, loss2, **kwargs):
        self.get_alpha(epoch, **kwargs)
        loss = self.alpha * loss1 + (1 - self.alpha) * loss2
        return loss
