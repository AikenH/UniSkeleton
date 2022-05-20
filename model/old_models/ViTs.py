# -*- coding: utf-8 -*-
"""
@ AikenH 2021 Transfomer

# reference: Microsoft Docs, 
# https://zhuanlan.zhihu.com/p/361366090?ivk_sa=1024320u
# https://github.com/berniwal/swin-transformer-pytorch/blob/master/swin_transformer_pytorch/swin_transformer.py
# https://cloud.tencent.com/developer/article/1819813
# https://www.jianshu.com/p/d50c1855e583
# type: Swin-T, Swin-S, Swin-B, Swin-L

#   when we training swin-transformer we will find that the GPU is not always occupy 100%
#   my program meet the bottleneck of data loading for ImageNet
#   should try some way to solve this.
"""
# install torchsummary in my local computer
# At the same time we test two method to define this module and compare which one is better
import torch
import numpy as np
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F

from torchstat import stat
from torchsummary import summary

# Some Blocks for ViT
# Patch Enbedding: 将原始图片裁成一个个window_size * window_size 的窗口大小然后线性嵌入
class PatchEmbed(nn.Module):
    """
    Patch Partition and Patch Merging,
    Downsampling for the Inputs,
    Using 1*1 abjust the channels,
    """

    def __init__(self, in_channels,out_channels,downscaling_factor):
        super(PatchEmbed, self).__init__()
        # get the scaling rate
        self.downscaling_factor = downscaling_factor
        
        # using unfold function to get the slidding windows
        self.patch_merge = nn.Unfold(downscaling_factor,stride=downscaling_factor,padding=0)
        
        # define the linear function
        self.linear = nn.Linear(in_channels * (downscaling_factor**2), out_channels)
        
    def forward(self,x):
        """
        input_shape: n,c,h,w 
        output_shape: n, h/f, w/f, c*f2 -> out_channels
        """
        n,_,h,w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor

        # Unfold 主要是为了保持原图中的位置信息
        x = self.patch_merge(x)

        # 对shape 进行还原是为了方便后续的图像中处理
        # 如果不在这里进行reshape的话，后面比较难还原shpe
        x = x.view(n,-1,new_h,new_w).permute(0,2,3,1)
        x = self.linear(x)

        return x

# Basic Block MLP
class m_MLP(nn.Module):
    def __init__(self,in_channels,hid_channels):
        super(m_MLP,self).__init__()

        self.layer1 = nn.Linear(in_channels, hid_channels)
        self.gelu = F.gelu
        self.layer2 = nn.Linear(hid_channels, in_channels)

    def forward(self,x):
        out = self.layer1(x)
        out = self.gelu(out)
        out = self.layer2(out)
        return out

# (shift) windows mulit-head self attention
# in this part we need to figure the diff from this with self-attention
# write self-attention ourself
def get_relative_pos(windows_s):
    indices = torch.tensor(
        np.array([[x,y] for x in range(windows_s) for y in range(windows_s)])
    )
    # using boardcast get the table of relative positions
    relative_pos = indices.unsqueeze(0) - indices.unsqueeze(1)
    return relative_pos

def generate_mask(window_size, displacement, upper_lower=False, left_right=False):
    # get the mask to calculate those area which is useful for shift

    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        win_s = window_size
        h,w = mask.shape

        mask = mask.view(win_s, h//win_s, win_s, w//win_s)
        # mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = mask.view(h,w)
        # mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask

class CyclicShift(nn.Module):
    # using torch.roll to shift the image(input)
    def __init__(self,displacement) :
        super(CyclicShift, self).__init__()
        self.displacement = displacement
    
    def forward(self,x):
        return torch.roll(x,shifts=(self.displacement,self.displacement),dims = (1,2))

class SWMulitHeadAttention(nn.Module):
    """
    Multi-Head-Attention module
        Dim_model is each block's.
    """
    def __init__(self, n_head, dim_model, dim_v,shifted, 
                    windows, relative_pos, drop_rate=0.1):
        super(SWMulitHeadAttention,self).__init__()

        # get the basic params for mulit-head attention
        self.n_head, self.dim_v = n_head, dim_v
        self.scale = dim_v ** 0.5
        # unique params for Swin
        self.shifted, self.windows = shifted, windows
        self.relative_pos = relative_pos
        # multi-head params
        # dim_v * n_head = dim_model
        self.m_query = nn.Linear(dim_model, dim_v * n_head, bias=False)
        self.m_keys = nn.Linear(dim_model, dim_v * n_head, bias=False)
        self.m_value = nn.Linear(dim_model, dim_v * n_head, bias=False)
        
        # shifted params
        if self.shifted:
            self.cycle_shift = CyclicShift(- windows//2)
            self.cycle_shift_re = CyclicShift( windows//2)
            self.lower_mask = nn.Parameter(generate_mask(windows,windows//2,upper_lower=True),
                                                requires_grad = False)
            self.right_mask = nn.Parameter(generate_mask(windows,windows//2,left_right=True),
                                                requires_grad = False)
        
        # get the position decoder for inputs (which is about the windows also)
        if self.relative_pos:
            self.relative_indice = get_relative_pos(windows) + windows -1 
            self.pos_embedding = nn.Parameter(torch.randn(2*windows-1, 2*windows-1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(windows**2, windows**2))

        # full connected section
        self.fc = nn.Linear(dim_v * n_head, dim_model)
        
        # dropout function
        self.dropout = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)

    def forward(self,x):
        """
        shape of input: batch,56,56,dim_model,
        data of heads: 3 
        shape of output: batch,56,56,dim_model,
        """
        # shift the input value
        if self.shifted:
            x = self.cycle_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.n_head

        # get (multi-head) q,k,v by input.
        # b,56,56,dim_k* n_head
        q,k,v = [fc(x) for fc in \
                    [self.m_query, self.m_keys, self.m_value]]

        # calculate the new shape of split to windows 
        window_h, window_w = n_h // self.windows, n_w // self.windows
        assert n_h % self.windows == 0 and n_w % self.windows ==0, \
            "the h,w should be fully devide by windows_size"
        # reshape qkv and separate the n_heads 
        # b,n_heads,56,56,dim_k
        # (windows): b,n_heads,(nw_h*nw_w),(windows)^2,dimk
        q,k,v = [temp.view(b,self.n_head,(window_h*window_w),(self.windows ** 2),self.dim_v) \
                    for temp in [q,k,v]]
        
        # calculate the scores
        # scores: b,n_head,56,56,56 
        # scores(windows): b,n_heads,(nw_h*nw_w),(windows)^2,(windows)^2
        scores = torch.matmul(q,k.permute(0,1,2,4,3)) * self.scale
        
        # add position embedding method
        if self.relative_pos:
            scores += self.pos_embedding[self.relative_indice[:,:,0].type(torch.long), 
                                        self.relative_indice[:,:,1].type(torch.long)]
        else:
            scores += self.pos_embedding
        
        # add the shift mask
        if self.shifted:
            scores[:,:,-window_w:] += self.lower_mask
            scores[:,:,window_w -1: window_w] += self.right_mask

        attn = scores.softmax(dim = -1)
        
        # out: b,n_heads,56,56,dim_k
        # out(windows): b,n_heads,(nw_h*nw_w),(windows)^2,dimk
        out = torch.matmul(attn,v)
        
        # reshape out (resume the origin shape of data)
        # b,56,56,dim_k* n_head
        out = out.view(b,n_h,n_w,-1)

        # get the real output
        out = self.fc(out)

        # shift to resume the map
        if self.shifted:
            out = self.cycle_shift_re(out)

        return out 
        
# Swin Transformer Block: which means we need even num of layers LN
class SwinBlock(nn.Module):
    def __init__(self, n_head, dim_model, dim_hid, dim_v,
                    shifted, windows, relative_pos,):
        super(SwinBlock, self).__init__()
        self.WMHA = SWMulitHeadAttention(n_head, dim_model, dim_v,shifted=shifted,
                                        windows = windows, relative_pos = relative_pos)

        self.LN1 = nn.LayerNorm(dim_model)
        self.MLP = m_MLP(dim_model,dim_hid)
        self.LN2 = nn.LayerNorm(dim_model)
    
    def forward(self,x):
        # Attention Block
        residual = x 
        x = self.LN1(x)
        x = self.WMHA(x)
        x += residual
        # MLP Block
        residual = x 
        x = self.LN2(x)
        x = self.MLP(x)
        x += residual

        return x 

# Stage Block for VIT:
class StageBlock(nn.Module):
    def __init__(self,in_channels, layers, dim_model,downscaling_factor,
                    n_head, dim_v, windows, relative_pos,):
        super(StageBlock, self).__init__()
        assert layers % 2 == 0, 'stage layer must be be even'
        
        self.patch_parition = PatchEmbed(in_channels, dim_model,downscaling_factor)
        self.layers = layers
        stages = []
        for i in range(layers // 2):
            stages += \
                [SwinBlock(n_head,dim_model,dim_model*4,dim_v,False,windows,relative_pos)]
            stages += \
                [SwinBlock(n_head,dim_model,dim_model*4,dim_v,True,windows,relative_pos)]
        self.Stage = nn.Sequential(*stages)

    def forward(self,x):
        x = self.patch_parition(x)
        x = self.Stage(x)
        x = x.permute(0,3,1,2)
        return x 

# Build a ViT
# In this framework will be swin-transformer 
class m_swin(nn.Module):
    def __init__(self,n_head, dim_model,layers, dim_v=32,num_classes=100,in_channels=3,
                    windows=7,downscaling_factor=(4,2,2,2),relative_pos=True):
        super(m_swin, self).__init__()
        self.stage1 = StageBlock(in_channels,layers[0],dim_model,downscaling_factor[0],
                                n_head[0],dim_v,windows,relative_pos)
        self.stage2 = StageBlock(dim_model,layers[1],dim_model*2,downscaling_factor[1],
                                n_head[1],dim_v,windows,relative_pos)
        self.stage3 = StageBlock(dim_model*2,layers[2],dim_model*4,downscaling_factor[2],
                                n_head[2],dim_v,windows,relative_pos)
        self.stage4 = StageBlock(dim_model*4,layers[3],dim_model*8,downscaling_factor[3],
                                n_head[3],dim_v,windows,relative_pos)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim_model*8),
            nn.Linear(dim_model*8, num_classes)
        )
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')

            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
        
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
    
    def forward(self,x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2,3])
        return self.mlp_head(x)

# Select one of those model
class SelectTrans():
    def __init__(self, num_classes, type='Swin-T',*args, **kwargs):
        super(SelectTrans, self).__init__()
        self.num_classes = num_classes
        self.type = type
    
    def GetModel(self):
        if self.type == 'Swin-T':
            params = {
                'n_head': (3,6,12,24),
                'dim_model': 96,
                'layers': (2,2,6,2),
                'num_classes': self.num_classes,
                'windows' : 7
            }
            model = m_swin(**params)

        elif self.type == 'Swin-S':
            params = {
                'n_head': (3,6,12,24),
                'dim_model': 96,
                'layers': (2,2,18,2),
                'num_classes': self.num_classes
            }
            model = m_swin(**params)

        elif self.type == 'Swin-B':
            params = {
                'n_head': (4,8,16,32),
                'dim_model': 96,
                'layers': (2,2,18,2),
                'num_classes': self.num_classes
            }
            model = m_swin(**params)

        elif self.type == 'Swin-L':
            params = {
                'n_head': (6,12,24,48),
                'dim_model': 96,
                'layers': (2,2,18,2),
                'num_classes': self.num_classes
            }
            model = m_swin(**params)
        else:
            raise NotImplementedError("add this in {}".format(self.__class__))
        return model

# Test module
def process():
     # using logging to print
    import logging
    logging.warning(__doc__)
    modelSelector = SelectTrans(100,'Swin-T')
    # select the model we want
    model = modelSelector.GetModel()
    
    # # get the randn data to test
    # tempdate = torch.randn(32,3,224,224)
    # out = model(tempdate)
    # logging.info(out)

    # visualize the 
    # stat(model, (3,224,224))
    summary(model.cuda(), input_size=(3,224,224),batch_size=1)

# Entry of test
if __name__ == '__main__':
   process()
