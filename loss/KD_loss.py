"""
@ Author: aikenhong 2022
@ Desc:
Design smoothing old label as a better distribution approximation 
to improve the kd loss.

@ Main idea:
The loss function consists of the following parts:

- (ce+mixup) : base-loss which means the specific task and using mixup to
                increase the loss and the complexity for the task
- (scl) : maintain task homogeneity to ensure feature extractor consistency 
- (kd-loss + label smoothing): Prevent old classes from being forgotten 
                                We may add the Feature constraints 

so we will using epoch to combine scl with ce(mix) to make sure the task (basic loss)
the combine kd to prevent forgotten.

@ what we need todo:
think about how to balance the new-class learning and the old forgetten
why thing like that, and what it supported to be, if we want to learn sth new.

@reference:
- https://github.com/seominseok0429/label-smoothing-visualization-pytorch?utm_source=catalyzex.com

"""
import torch
from torch import nn
import torch.nn.functional as F

# def kd_ce(outputs, targets,exp=1.0, size_average=True, eps=1e-5):
#     """Calculates cross-entropy with temperature scaling"""
#     out = torch.nn.functional.softmax(outputs, dim=1)
#     tar = torch.nn.functional.softmax(targets, dim=1)
#     if exp != 1:
#         out = out.pow(exp)
#         out = out / out.sum(1).view(-1, 1).expand_as(out)
#         tar = tar.pow(exp)
#         tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
#     out = out + eps / out.size(1)
#     out = out / out.sum(1).view(-1, 1).expand_as(out)
#     ce = -(tar * out.log()).sum(1)
#     if size_average:
#         ce = ce.mean()
#     return ce
class KD(nn.Module):
    def __init__(self, factor=1.0, temperature=2, criterion=None,**kwargs):
        super(KD, self).__init__()
        self.factor = factor
        self.T = temperature
        self.criterion = criterion

    def forward(self, new_pred, old_pred, target, basic_loss=None,**kwargs):
        num_old = old_pred.size(1)

        if basic_loss is None: 
            if self.criterion is not None: 
                basic_loss = self.criterion(new_pred, target)
            else: 
                print("Warning: Do not provide the basic loss and the Basic Loss!")
                basic_loss = 0 
        
        kd_loss = F.binary_cross_entropy(F.softmax(new_pred[:,:num_old]/self.T, dim=1),
                                            F.softmax(old_pred.detach()/self.T, dim=1))
        
        loss = self.factor * kd_loss + (1- self.factor) * basic_loss

        return loss
        
def kdloss(new_pred, old_pred, targets, criterion,
        basic_loss=None, temporature=None,):
    """
    args:
        criterion: loss function
        new_pred: prediction of new model
        old_pred: prediction of old model
        targets: target
        num_new: number of new cls
        num_old: number of old cls
        basic_loss: basic loss result
        temporature: temperature
    
    Return:
        Knowledge Distill loss
    """
    num_new = new_pred.size(1)
    num_old = old_pred.size(1)
    # lam = num_old / num_new
    lam = 0.9

    # get the basic loss
    if basic_loss is None:
        basic_loss = criterion(new_pred, targets)
        
    kd_loss = F.binary_cross_entropy(F.softmax(new_pred[:,:num_old]/2, dim=1),
                                    F.softmax(old_pred.detach()/2, dim=1))

    # calculate the total loss
    # loss = lam * basic_loss + (1 - lam) * kd_loss
    loss = lam * kd_loss + (1 - lam) * basic_loss
    return loss

class LabelSmoothingKD(nn.Module):
    def __init__(self, lam=0.8,smoothing=0.1, combiner=None, epoches=None, **kwargs):
        super(LabelSmoothingKD, self).__init__()
        """
        init those parameters of kd loss calculator
        """
        self.lam = lam
        self.smoothing = smoothing
        self.combiner = combiner
        self.epoches = epoches


    def forward(self, new_pred, old_pred, targets, criterion,
            basic_loss=None, temporature=2, epoch=None,smoothing=None,
            **kwargs):
        """
        calculate smoothing-kd loss(lwf), and return the loss
        support more distill loss in the future.
        """
        # smoothen the old prediction and get the better approximation
        # NOTE: record this way to write a choose
        smoothing = smoothing or self.smoothing
        old_pred = self._make_smoothen(old_pred, new_pred, smoothing)

        # calculate the basic loss
        if basic_loss is None: basic_loss = criterion(new_pred, targets)

        # update the lam
        self.lam = self._update_lambda(epoch, **kwargs)

        # calculate the kd loss
        lwf = kd_ce(new_pred, old_pred, exp=1/temporature)
        loss = self.lam * lwf + (1 - self.lam) * basic_loss

        return loss

    def _update_lambda(self, epoch, **kwargs):
        # update the lambda by epoch
        if self.combiner == 'fix' or self.combiner is None:
            return self.lam

        assert self.epoches != None, "epoches should be given"

        if self.combiner == 'linear':
            lam = (epoch -1) / self.epoches

        elif self.combiner == 'parabolic':
            lam = ((epoch -1 )/ self.epoches)**2

        elif self.combiner == 'separate':
            threshold = kwargs.get('threshold', 120)
            lam = 1 if threshold< epoch else 0

        elif self.combiner == 'cosine':
            import math
            lam = math.cos((epoch-1)/self.epoches * math.pi/2)

        elif self.combiner == 'abort': lam = 0
        else: raise NotImplementedError("combiner should be add")

        return lam

    def _make_smoothen(self, x, y, smoothing):
        """ x is the original prediction,
        y should be the target prediction """
        if smoothing == 0:  return x

        assert x.shape != y.shape, "the shape of x and y should be different"
        extra_len = y.size(1) - x.size(1)

        x = x * (1-smoothing)
        extra_len = torch.tensor([smoothing/extra_len] * extra_len).expand(x.size(0), -1).cuda()
        x = torch.cat([x, extra_len], dim=1)

        return x

from util.utils import make_onehot_array
def iCaRL_loss(predict, old_predict, target, num_classes=100, factor=0.7, basic_loss=None, *args, **kwargs):
    """
    ICaRL loss design, which replace the ce with the distill loss
    Combine the Old prediction and the New Labels.
    
    I think this is better than the Smooth Loss.

    And this one should intergate with the ce
    """
    # calculate the ce loss
    if basic_loss is None: 
        if kwargs.get('criterion'):
            basic_loss = kwargs['criterion'](predict, target)

    # calculate the distill loss
    target = make_onehot_array(num_classes, target).cuda()
    with torch.no_grad():
        old_predict = torch.sigmoid(old_predict)
    old_task_size = old_predict.shape[1]
    target[...,:old_task_size] = old_predict

    loss = factor * F.binary_cross_entropy_with_logits(predict, target)
    if basic_loss is not None: loss += (1-factor) * basic_loss

    return loss

class ICARL(nn.Module):
    def __init__(self, num_classes=100, factor=0.7, criterion=None, *args, **kwargs):
        super(ICARL, self).__init__()
        self.factor = factor
        self.num_classes = num_classes
        self.criterion = criterion
    
    def forward(self, predict, old_predict, target, basic_loss=None, **kwargs):
        # calculate the ce loss
        if basic_loss is None: 
            if self.criterion is not None: 
                basic_loss = self.criterion(predict, target)
            else: 
                print("Warning: Do not provide the basic loss and the Basic Loss!")
                basic_loss = 0
        
        # calculate the kd loss
        target = make_onehot_array(self.num_classes, target).cuda()
        with torch.no_grad():
            old_predict = torch.sigmoid(old_predict)
        old_task_size = old_predict.shape[1]
        target[...,:old_task_size] = old_predict

        loss = self.factor * F.binary_cross_entropy_with_logits(predict, target)
        loss += (1-self.factor) * basic_loss
        
        return loss
