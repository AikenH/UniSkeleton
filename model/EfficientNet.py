"""
@AikenH 2021 
# Efficient Net Model here
# Type: b0-b7
# paper: efficientnet,
# reference: 
# best resolution: [224,240,260,300,380,456,528,600] 
"""

import torch 
from torch import nn 
from collections import OrderedDict
from torch.nn import functional as F 

from torchstat import stat 
from torchsummary import summary

# from layers.convolution import SeparableConv2d
# from layers.new_activation import Swish

class SeparableConv2d(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=1,stride=1,padding=0,bias=False):
        super(SeparableConv2d,self).__init__()

        self.depthwise = nn.Conv2d(in_channel,in_channel,kernel,stride,padding,
                                groups=in_channel,bias=False)
        # self.pointwise = nn.Conv2d(in_channel,out_channel,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.depthwise(x)
        # x = self.pointwise(x)
        return x

class Swish(nn.Module):
    def __init__(self,beta=1):
        super(Swish,self).__init__()
        self.beta = beta

    def forward(self,x):
        # x  *= torch.  (self.beta * x)
        return x * torch.sigmoid(x)

# 最基本layer
class ConvN_N(nn.Module):
    def __init__(self,IN_CHANNEL,OUT_CHANNEL,kernel):
        super(ConvN_N, self).__init__()
        pad = (int)(kernel-1)//2
        self.convn_n = nn.Sequential(OrderedDict([ 
            ('conv11',nn.Conv2d(IN_CHANNEL,OUT_CHANNEL,kernel,padding=pad,bias=False)),
            ('bn11',nn.BatchNorm2d(OUT_CHANNEL))
        ]))

    def forward(self,x):
        return self.convn_n(x)

class DWConvN_N(nn.Module):
    def __init__(self,IN_CHANNEL,OUT_CHANNEL,kernel,stride=1):
        super(DWConvN_N, self).__init__()
        # 模拟padding = same的操作
        pad = (int)((kernel-1) // 2) 
        self.module1 = nn.Sequential(OrderedDict([
            ('depthwiseConv',SeparableConv2d(IN_CHANNEL,OUT_CHANNEL,
                            kernel=kernel,stride=stride,padding=pad)),
            ('bn',nn.BatchNorm2d(OUT_CHANNEL)),
            ('Swish',Swish())
        ]))
        
        
    def forward(self, x):
        out = self.module1(x)
        return out 

class SE(nn.Module):
    def __init__(self,w_in,w_se):
        super(SE, self).__init__()
        w_se = (int)(w_se)
        self.seblock = nn.Sequential(OrderedDict([ 
            ('avgpool', nn.AdaptiveAvgPool2d((1,1))),
            ('convfc1', nn.Conv2d(w_in,w_se,1,bias=True)),
            ('Swish',Swish()),
            ('convfc2', nn.Conv2d(w_se,w_in,1,bias=True))
        ]))
        
    def forward(self,x):
        return x * torch.sigmoid(self.seblock(x))

""" class SE(nn.Module):
    '''Squeeze-and-Excitation block with Swish.'''

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels,
                             kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_channels, in_channels,
                             kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = Swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out """

# 基本的三个大单元
# module3
class MBConvBlock(nn.Module):
    def __init__(self,IN_CHANNEL,kernel,stride=1,se_ratio=0.25):
        super(MBConvBlock,self).__init__()
        self.layer1 = ConvN_N(IN_CHANNEL,6*IN_CHANNEL,1)
        self.swish = Swish()
        self.layer2 = DWConvN_N(6*IN_CHANNEL,6*IN_CHANNEL,kernel,stride=1)
        self.SE = SE(6*IN_CHANNEL,IN_CHANNEL*se_ratio)
        self.convf = ConvN_N(6*IN_CHANNEL,IN_CHANNEL,1)
        self.dropout = nn.Dropout(0)

    def forward(self,x):
        out = self.layer1(x)
        out = self.swish(out)
        out = self.layer2(out)
        out = self.SE(out)
        out = self.convf(out)
        out = self.dropout(out)

        return x + out 

# module1 
class SepConv(nn.Module):
    def __init__(self,IN_CHANNEL,OUT_CHANNEL,kernel=3,stride=1,se_ratio=0.25):
        super(SepConv,self).__init__()
        self.DWC = DWConvN_N(IN_CHANNEL,IN_CHANNEL,kernel,stride)
        self.SE = SE(IN_CHANNEL,IN_CHANNEL*se_ratio)
        self.Final = ConvN_N(IN_CHANNEL,OUT_CHANNEL,1)
        # self.convf = nn.Conv2d(IN_CHANNEL,OUT_CHANNEL,1,bias=True)
        # self.bnf = nn.BatchNorm2d(OUT_CHANNEL)
    
    def forward(self,x):
        out = self.DWC(x)
        out = self.SE(out)
        out = self.Final(out)
        # out = self.convf(out)
        # out = self.bnf(out)
        return out

# module2
class MBConv(nn.Module):
    def __init__(self,IN_CHANNEL,OUT_CHANNEL,kernel,stride=1,se_ratio=0.25):
        super(MBConv, self).__init__()
        self.layer1 = ConvN_N(IN_CHANNEL,6*IN_CHANNEL,1)
        self.swish = Swish()
        self.layer2 = DWConvN_N(6*IN_CHANNEL,6*IN_CHANNEL,kernel,stride)
        self.SE = SE(6*IN_CHANNEL,IN_CHANNEL*se_ratio)
        self.convf = ConvN_N(6*IN_CHANNEL,OUT_CHANNEL,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.swish(out)
        out = self.layer2(out)
        out = self.SE(out)
        out = self.convf(out)

        return out

# # MBConv 作为一个单元整合的模块来写，主要由3种block和相应的堆叠系数来构成
""" class MBBlock(nn.Module):
    def __init__(self,):
        super(MBBlock, self).__init__()
    
    def _make_layer(self,):
        ... 
        
    def forward(self,x):
        ... """

class EfficientNetModel(nn.Module):
    def __init__(self, CHANNELS, LAYERS, IN_CHANNEL=3, NUM_CLASSES=1000 ,STEM_W=32, HEAD_W=1280, ac_layer=None):
        super(EfficientNetModel, self).__init__()
        KERNELS = [3,3,5,3,5,5,3]
        STRIDES = [1,2,2,2,1,2,1]
        # process of data: rescaling- normalize- zeropadding which should be done on dataloader
        
        # STEM layer: 考虑可定制的话，实际上可能可以独立成一个模块
        # ATTENTION: 第一个stride最好还是设为一，这种情况下才能更好的捕捉图片的信息，不会导致一些训练准确率特别低的问题
        self.stem = nn.Sequential(OrderedDict([
            ('STEM_conv', nn.Conv2d(IN_CHANNEL,STEM_W,3,stride=1,padding=1,bias=False)),
            ('STEM_bn',nn.BatchNorm2d(STEM_W)),
            ('STEM_ac',Swish())
        ]))
        
        # MBConv :3 type
        self.blocks = nn.Sequential()
        for i,nums in enumerate(LAYERS):
            if i == 0:
                self.blocks.add_module("Block1",self._make_layer(nums,SepConv,MBConvBlock,
                                    STEM_W,CHANNELS[i],KERNELS[i],STRIDES[i]))
            else:
                self.blocks.add_module("Block{}".format(i+1),self._make_layer(nums,MBConv,MBConvBlock,
                                    CHANNELS[i-1],CHANNELS[i],KERNELS[i],STRIDES[i]))
        
        # Final layer
        self.final_layer = nn.Sequential(OrderedDict([
            ('convf',nn.Conv2d(CHANNELS[-1],HEAD_W,kernel_size=1,stride=1)),
            ('bnf',nn.BatchNorm2d(HEAD_W)),
            ('acf',Swish())
        ]))
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)

        # make it optional block
        self.fc = nn.Linear(HEAD_W,NUM_CLASSES,bias=True) 
        

        # # Init the parameters
        # for m in self.modules():
        #     if isinstance(m,(nn.Conv2d,nn.Linear)):
        #         nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

        #     if isinstance(m,nn.BatchNorm2d):
        #         nn.init.constant_(m.weight,1)
        #         nn.init.constant_(m.bias,0)
    
    def forward(self,x):
        out = self.stem(x)
        out = self.blocks(out)
        out = self.final_layer(out)
        # out = self.global_pool(out)
        # out = out.view(out.size(0),-1)
        # out = self.dropout(out)

        # out = self.fc(out)

        return out 
                
    def _make_layer(self,num_layers:int,block_basic,block_re,in_cha, out_cha, kernel, stride):
        layers = []
        layers.append(block_basic(in_cha,out_cha,kernel,stride))

        for times in range(num_layers-1):
            layers.append(block_re(out_cha,kernel,stride))
        
        net = nn.Sequential(*layers)
        return net
            
# Test model u write here
class ChooseEfficientNet():
    # best resolution: [224,240,260,300,380,456,528,600]
    def __init__(self,num_classes,type='EfficientNetb0'):
        super(ChooseEfficientNet,self).__init__()
        self.num_classes = num_classes
        self.type = type 

    def GetModel(self):
        if self.type == 'EfficientNetb0':
            params = {
                'CHANNELS': [16,24,40,80,112,192,320],
                # 'LAYERS': [1,2,2,3,3,4,1],
                'LAYERS': [1,2,2,3,3,4,1],
                'STEM_W': 32,
                'HEAD_W': 1280,
                'NUM_CLASSES': self.num_classes
            }
            model = EfficientNetModel(**params)

        elif self.type == 'EfficientNetb1':
            params = {
                'CHANNELS':[16,24,40,80,112,192,320],
                'LAYERS':[2, 3, 3, 4, 4, 5, 2],
                'STEM_W': 32,
                'HEAD_W': 1280,
                'NUM_CLASSES': self.num_classes
            }
            model = EfficientNetModel(**params)

        elif self.type == 'EfficientNetb2':
            params = {
                'CHANNELS':[16, 24, 48, 88, 120, 208, 352],
                'LAYERS':[2, 3, 3, 4, 4, 5, 2],
                'STEM_W': 32,
                'HEAD_W': 1408,
                'NUM_CLASSES': self.num_classes
            }
            model = EfficientNetModel(**params)
        
        elif self.type == 'EfficientNetb3':
            params = {
                'CHANNELS':[24, 32, 48, 96, 136, 232, 384],
                'LAYERS':[2, 3, 3, 5, 5, 6, 2],
                'STEM_W': 40, 
                'HEAD_W': 1536,
                'NUM_CLASSES': self.num_classes
            }
            model = EfficientNetModel(**params)
        
        elif self.type == 'EfficientNetb4':
            params = {
                'CHANNELS':[24, 32, 56, 112, 160, 272, 448],
                'LAYERS':[2, 4, 4, 6, 6, 8, 2],
                'STEM_W': 48, 
                'HEAD_W': 1792,
                'NUM_CLASSES': self.num_classes
            }
            model = EfficientNetModel(**params)
        
        elif self.type == 'EfficientNetb5':
            params = {
                'CHANNELS':[24, 40, 64, 128, 176, 304, 512],
                'LAYERS':[3, 5, 5, 7, 7, 9, 3],
                'STEM_W': 48,
                'HEAD_W': 2048,
                'NUM_CLASSES': self.num_classes
            }
            model = EfficientNetModel(**params)

        elif self.type == 'EfficientNetb6':
            params = {
                'CHANNELS':[32, 40, 72, 144, 200, 344, 576],
                'LAYERS':[3, 6, 6, 8, 8, 11, 3],
                'STEM_W': 56, 
                'HEAD_W': 2304,
                'NUM_CLASSES': self.num_classes
            }
            model = EfficientNetModel(**params)
        
        elif self.type == 'EfficientNetb7':
            params = {
                'CHANNELS':[32, 48, 80, 160, 224, 384, 640],
                'LAYERS':[4, 7, 7, 10, 10, 13, 4],
                'STEM_W': 64, 
                'HEAD_W': 2560,
                'NUM_CLASSES': self.num_classes
            }
            model = EfficientNetModel(**params)
        
        else:
            raise NotImplementedError("add this in {}".format(self.__class__))

        # from model.backupEffic import EfficientNetB0
        # model = EfficientNetB0()

        return model
    
if __name__ == "__main__":
    model_s = ChooseEfficientNet(100,'EfficientNetb8')
    model = model_s.GetModel()
    # print(model)
    # tempdata = torch.rand(3,3,224,224)

    # stat(model, (3,224,224))
    summary(model.cuda(), input_size=(3,48,48),batch_size=1) 