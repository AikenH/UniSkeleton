"""@AikenH 2021 universal utils
some helpful methods will be writed in here.
"""

import torch
import numpy as np
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

def flatten_batch(batches: torch.Tensor):
    """receive a batches of datas and return a list of numpy datas"""
    res = []
    try:
        batches = batches.cpu().numpy()
    except AttributeError:
        print('batches is a {}, we do not need to convert it to numpy'.format(
            type(batches)))

    for i in range(batches.shape[0]):
        res.append(batches[i])
    return res
    
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
        return res, res>old

def new_min_flag(old,new):
    if old is None:
        return new,False 
    else:
        res = min(old,new)
        return res, res<old

def make_onehot_single(num,index):
    # BTW：scatter方法也能生成one-hot
    onehot = torch.zeros(num)
    onehot[index] = 1.0
    
    return onehot

def try_next(dataitertorA,dataloadeA):
    try:
        imgs,labels = next(dataitertorA)

    except StopIteration:
        dataitertorA = iter(dataloadeA)
        imgs,labels = next(dataitertorA)

    return imgs,labels

def make_onehot_array(width,target):
    '''generate onthot matrix from label 
    width：类别数 
    target：具体的labeldata'''
    try:
        length = len(target.view(-1,1))
    except ValueError:
        print('the type of target is {} '.format(type(target)))
        print(target)
        raise Exception('break down')
    onehot = torch.zeros(length, width).cuda().scatter_(1,target.view(-1,1),1)
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

def remap_index(labelist,start=0):
    """
    Map the old index to new, avoid the label be overflow
    the Start should be setting by the new cls nums
    """
    # Note the the label should be start from 0 dont be fool by the ans in the net
    # get the random dict of labels
    old_label_set = np.unique(labelist)
    np.random.shuffle(old_label_set)
    
    # getting the mapping dict
    mappingdic = {}
    for value, key in enumerate(old_label_set):
        mappingdic[key] = value + start
    
    # mapping the labellist
    res = []
    for i in range(len(labelist)):
        res.append(mappingdic[labelist[i]])
    
    return res

def mapping_order_list(ordered_list, old_targets):
    """
    the second way to mapping the label
    the orderlist's index is what we want to assign the labels
    """
    mappingdic = {}
    # build the dict by the list which index already in order
    for value,key in enumerate(ordered_list):
        mappingdic[key] = value
    
    # mapping the targets
    res = []
    for i in range(len(old_targets)):
        res.append(mappingdic[old_targets[i]])
    
    return res

class EarlyStopping():
    def __init__(self, early_stop=True ,patience=10, mode='max',verbose=False,logger=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.mode = mode
        self.best_score = float('inf') if mode=='min' else float('-inf')
        self.early_stop = early_stop
        self.logger = logger
        self.flag = False
    
    def step(self, score):
        """
        Stop Training when return TRUE
        """
        self.best_score,self.flag = \
            new_min_flag(self.best_score,score) if self.mode=='min' else new_max_flag(self.best_score,score)
        self.counter = 0 if self.flag else self.counter + 1
        
        if self.counter >= self.patience:
            if self.logger is not None and self.verbose:
                self.logger.info('Achieved the patience of ES, STOP TRAINING!')
            return True        
        return False
        
