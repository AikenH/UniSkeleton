"""@AikenH 2021 universal utils
some helpful methods will be writed in here.
"""

import torch
from torchstat import stat
from thop import profile, clever_format

# UPDATE：使用thop进行参数和相应的计算复杂度分析,但是还要考虑一些自定义规则之类的，如果自己编写的layer有特殊情况
# @ https://github.com/Lyken17/pytorch-OpCounter 
def FLOPS_Params(model,input):
    try:
        macs,params = profile(model, inputs = (input,))
        macs,params = clever_format([macs,params], "%.3f")
        return macs,params
    except:
        stat(model,(3,224,224))
        return 
 
# old：下面这部分已经相应的库来替代
def count_params(model, is_grad = True):
    """ count the num of params in models
    Args:
        model;
        is_grad (bool, optional): [true: all the params need grad
                                   False: all parameters]. 
        Defaults to True.
    Returns:
        [int]: [the num of params in this model]
    """
    if is_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

# 给old值设定一个None的情况同时返回一个是否是True的判断
# 这里可能会减慢速度，后续可以考虑改进这个模块

def new_max_flag(old,new):
    if old is None:
        return new,False
    else:
        res = max(old,new)
        return res, res<=old

def new_min_flag(old,new):
    if old is None:
        return new,False 
    else:
        res = min(old,new)
        return res, res>=old

def make_onehot_single(num,index):
    # BTW：scatter方法也能生成one-hot
    onehot = torch.zeros(num)
    onehot[index] = 1.0
    
    return onehot

def make_onehot_array(width,target):
    '''根据label生成onehot矩阵。 
    width：类别数 target：具体的labeldata'''
    try:
        length = len(target.view(-1,1))
    except ValueError:
        print('the type of target is {} '.format(type(target)))
        print(target)
        raise Exception('break down')
    onehot = torch.zeros(length, width).scatter_(1,target.view(-1,1),1)
    # onehot = torch.zeros(length, width).scatter_(1,target.view(-1,1).float(),1)

    return onehot

def select_n_random(data, label, n=100):
    """
    Docs of Pytorch
    Selects n random datapoints and their corresponding labels from a dataset
    """
    assert len(data) == len(label)
    perm = torch.randperm(len(data))
    return data[perm][:n], label[perm][:n]