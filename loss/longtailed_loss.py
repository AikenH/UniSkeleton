"""
@AikenHong2021
This file is used to intergrate the long-tailed loss design
Which is using to align the bias.

"""
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class DisAlignLoss(nn.Module):
    """
    Paper: Distribution Alignment: A Uniﬁed Framework for Long-tail Visual Recognition (CVPR 2021)
    arXiv: https://arxiv.org/abs/2103.16370
    Source Code: https://github.com/Megvii-BaseDetection/cvpods/blob/master/cvpods/modeling/losses/grw_loss.py
    """
    def __init__(self, cls_num_list=None, p=1.5):
        super().__init__()

        if cls_num_list is None:
            self.m_list = None
            self.per_cls_weights = None
        else:
            self.m_list = torch.from_numpy(np.array(cls_num_list))

            self.per_cls_weights = self.m_list / self.m_list.sum()  # r_c in paper
            self.per_cls_weights = 1.0 / self.per_cls_weights
            self.per_cls_weights = self.per_cls_weights ** p
            self.per_cls_weigths = self.per_cls_weights / self.per_cls_weights.sum() * len(cls_num_list)

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)

        return self

    def forward(self, output_logits, target):
        return F.cross_entropy(output_logits, target, weight=self.per_cls_weights)