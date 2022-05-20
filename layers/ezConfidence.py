""" 
@AikenH 2021
# 汇总一些简单的置信度评估方法，用于可视化输出
# 按照每个batch作为标准来进行平均输出相应的置信度
"""

import numpy as np 
import matplotlib.pyplot as plt 

import torch 

def leastConfi(pred,mapfuc=None,avg=True):
    """以预测值中的最大概率值作为预测的置信度，
    区间可以划分为：[1/num_class,1] -> [0,1]

    Args:
        pred (tensor): 对应的预测值
    """
    pred_soft = torch.softmax(pred,dim=1)
    num_class = pred_soft.size(1)
    maxV, _ = torch.max(pred_soft,dim=1)
    meanV = 1./num_class
    
    BConfidence = None 
    if mapfuc is None:
        # 比较直观简单的方法就是均匀分布预测
        # 考虑维度扩散的问题
        BConfidence = (maxV - meanV)/(1 - meanV)
    else:
        ...
    
    if avg:
        return BConfidence.mean()
    else:
        return BConfidence.mean(), BConfidence

def EntropyConfi(pred,isAvg=True):
    """按照熵值来分析置信度（信息的不稳定程度）
    """
    logpred = torch.log2(pred)
    entropy = torch.mul(pred,logpred)
    if isAvg:
        entropy = entropy.sum(dim=1).mean()
    return entropy