import copy
import torch
import torch.nn as nn
import math

# using this module as the clfs' last layer, and we need to
class Causal_Norm_Classifier(nn.Module):
    def __init__(self, num_cls=100, in_dim=2048, use_effect=True, num_head=2, 
                    tau=16.0, alpha=1.5, gamma=0.03125, *args, **kwargs):
        super(Causal_Norm_Classifier, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_cls, in_dim).cuda(), requires_grad=True)
        self.scale = tau / num_head   # 16.0 / num_head
        self.norm_scale = gamma       # 1.0 / 32.0
        self.alpha = alpha            # 3.0
        self.num_head = num_head
        self.head_dim = in_dim // num_head
        self.use_effect = use_effect
        self.reset_parameters(self.weight)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1)) # 标准差
        weight.data.uniform_(-stdv, stdv) # 重新初始化，用均匀分布的数据来填充weight

    def forward(self, x, embed=None):
        if x.dim() >= 3: 
            x = self.avgpool(x).view(x.size(0), -1)
        # calculate capsule normalized feature vector and predict
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)
        normed_x = self.multi_head_call(self.l2_norm, x)
        y = torch.mm(normed_x * self.scale, normed_w.t())

        # remove the effect of confounder c during test
        if (not self.training) and self.use_effect and (embed is not None):
            # if embed is None: raise ValueError('embed is None in test mode')
            self.embed = torch.from_numpy(embed).view(1, -1).to(x.device)
            normed_c = self.multi_head_call(self.l2_norm, self.embed)
            head_dim = x.shape[1] // self.num_head
            x_list = torch.split(normed_x, head_dim, dim=1)
            c_list = torch.split(normed_c, head_dim, dim=1)
            w_list = torch.split(normed_w, head_dim, dim=1)
            output = []

            for nx, nc, nw in zip(x_list, c_list, w_list):
                cos_val, sin_val = self.get_cos_sin(nx, nc)
                y0 = torch.mm((nx -  cos_val * self.alpha * nc) * self.scale, nw.t())
                output.append(y0)
            y = sum(output)
            
        return y

    def get_cos_sin(self, x, y):
        cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
        sin_val = (1 - cos_val * cos_val).sqrt()
        return cos_val, sin_val

    #multi-head-setting part 
    def multi_head_call(self, func, x, weight=None):
        assert len(x.shape) == 2
        x_list = torch.split(x, self.head_dim, dim=1) # 相当于我们将从特征维度（而非样本维度，将数据根据head划分成了块）
        if weight:
            y_list = [func(item, weight) for item in x_list]
        else:
            y_list = [func(item) for item in x_list]
        assert len(x_list) == self.num_head
        assert len(y_list) == self.num_head
        return torch.cat(y_list, dim=1)

    def l2_norm(self, x):
        # 使用l2 normalization（标准化） 范数来对x进行处理
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

    def capsule_norm(self, x):
        # 使用
        norm= torch.norm(x.clone(), 2, 1, keepdim=True)
        normed_x = (norm / (1 + norm)) * (x / norm)
        return normed_x

    def causal_norm(self, x, weight):
        # 自定义的因果分析范数，来对x进行标准化去除不均匀分布的影响
        norm= torch.norm(x, 2, 1, keepdim=True)
        normed_x = x / (norm + weight)
        return normed_x
    
    def _expand_dim(self, num_cls, re_init=True, hidden_layers=None, *args,**kwargs):
        # save the old parameters 
        old_weights = copy.deepcopy(self.weight.data)
        feat_dim = old_weights.shape[1]
        
        # reinit a new weight
        self.weight = nn.Parameter(torch.Tensor(num_cls, feat_dim).cuda(), requires_grad=True)
        if re_init: self.reset_parameters(self.weight)

        # pass the old parameters to the new weight
        self.weight.data[:old_weights.shape[0]] = old_weights

        return None
    