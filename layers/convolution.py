"""
@AikenH 2021 convolution kernel 
# reference: xception
# https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py 
# 特性主要来自于conv中的group参数，通过分组来实现这种分离的效果
"""
from torch import nn 

# depth-wise convolution kernal here
class SeparableConv2d(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=1,stride=1,padding=1,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channel,in_channel,kernel,stride,padding,dilation,
                                groups=in_channel,bias=False)
        # self.pointwise = nn.Conv2d(in_channel,out_channel,1,1,0,groups=1,bias=bias)
    
    def forward(self,x):
        out = self.conv1(x)
        # out = self.pointwise(out)
        return out

