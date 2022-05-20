"""
@AikenH 2021 ResNet 
# type: 18 34 50 101 152 (which can decide the num_cls by myself)
# reference：PyTorch Document
"""
# install torchsummary in my local computer
# At the same time we test two method to define this module and compare which one is better
import torch
from torch import nn
from collections import OrderedDict

from torchstat import stat
from torchsummary import summary

# this is the conv block,which have two type: bottleneck basicblokc
# and two connect methods for each one 
class Basicblock(nn.Module):
    expandFactor = 1
    def __init__(self,inplane, outplane, stride=1, Downsample=None,*args,**kwargs):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Conv2d(inplane, outplane,3,stride,padding=1)
        self.bn1 = nn.BatchNorm2d(outplane)
        self.relu1 = nn.ReLU(inplace=True)
        
        # this step should not use stride here
        self.conv2 = nn.Conv2d(outplane, outplane,3,padding=1)
        self.bn2 = nn.BatchNorm2d(outplane)
        self.relu2 = nn.ReLU(inplace = True)

        self.downsample = Downsample
        self.stride = stride

    def forward(self,x):
        resudual = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        # NOTE: the bn's location and the resudual connect should be in attention
        out = self.bn2(out) 
        
        # consider whether we need downsample 
        if self.downsample is not None:
            resudual = self.downsample(x)
        
        out += resudual
        out = self.relu2(out)

        return out

class Bottleneck(nn.Module):
    expandFactor = 4
    def __init__(self,inplane,outplane,stride=1,Downsample=None,*args,**kwargs):
        super(Bottleneck, self).__init__()
        self.expandFactor = Bottleneck.expandFactor

        self.block1 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(inplane,outplane,1)),
            ("bn1", nn.BatchNorm2d(outplane)),
            ("relu1", nn.ReLU(inplace= True))
        ]))

        self.block2 = nn.Sequential(OrderedDict([
            ("conv2", nn.Conv2d(outplane,outplane,3,stride,padding=1)),
            ("bn2", nn.BatchNorm2d(outplane)),
            ("relu2", nn.ReLU(inplace= True))
        ]))

        out_dim = outplane * self.expandFactor
        self.conv3 = nn.Conv2d(outplane,out_dim,1)
        self.bn3 = nn.BatchNorm2d(out_dim)
        self.relu3 = nn.ReLU(inplace= True)

        self.downsample = Downsample
        self.stride = stride

    def forward(self,x):
        resudual = x 

        out = self.block1(x)
        out = self.block2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            resudual = self.downsample(x)
        
        out += resudual
        out = self.relu3(out)

        return out

class m_ResNet(nn.Module):
    def __init__(self,block,num_classes,num_blocks,*args,**kwargs):
        super(m_ResNet, self).__init__()
        # baisc unit which have default parameters
        self.block0 = nn.Sequential(OrderedDict([
            ('conv0',nn.Conv2d(3,64,7,stride=2,padding=3)),
            ('bn0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace= True)),
            # ('maxpooling', nn.MaxPool2d(3,stride=2,padding=1))
        ]))
        #  make layer by block we choose.
        self.block1 = self._make_layer(block,64,64,num_blocks[0])
        self.block2 = self._make_layer(block,64*block.expandFactor,128,num_blocks[1],stride=2)
        self.block3 = self._make_layer(block,128*block.expandFactor,256,num_blocks[2],stride=2)
        self.block4 = self._make_layer(block,256*block.expandFactor,512,num_blocks[3],stride=2)

        # fc and avg pooling 
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expandFactor,num_classes)

        # Using specific classifer
        if kwargs.get('classifier'):
            self.fc = kwargs['classifier'](512*block.expandFactor,num_classes)

        self.relu = nn.ReLU(inplace= True)
        

        # # init the model parameters for all the layers 
        # # 这里是否需要进行一个超惨的选择，也就是初始化方法该怎么去决定
        # for m in self.modules():
        #     if isinstance(m,nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
        #     if isinstance(m,nn.BatchNorm2d):
        #         nn.init.constant_(m.weight,1)
        #         nn.init.constant_(m.bias,0)
            
    def _make_layer(self,block,inplane,outplane,num_block,stride=1,*args,**kwargs):
        # 这里需要判断是否需要downsample，
        # 1. 是否需要虚连接，只有在第一个layer可能是需要的
        # 2. 是否图片的size不一致（由于padding 策略应该都是same，所以只考虑stride）
        downsample = None
        
        if inplane != outplane*block.expandFactor:
            downsample = nn.Sequential(nn.Conv2d(inplane,outplane*block.expandFactor,1,
                                        stride=stride,bias=False),
                                        nn.BatchNorm2d(outplane*block.expandFactor))
            
        # 添加相应的layer，
        # NOTE: the first layer is specific, becus the input dim is different
        # and the stride and the downsample is only in the first layer
        layers = []
        layers.append(block(inplane,outplane,stride=stride,Downsample=downsample))
        inplane = outplane* block.expandFactor
        for i in range(1,num_block):
            layers.append(block(inplane,outplane))
        
        net = nn.Sequential(*layers)
        return net
    
    def forward(self,x):
        out = self.block0(x)
        
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        
        # feature = out 
        
        # out = self.avgpool(out)
        # # resize 到一个维度
        # out = out.view(out.size(0),-1)
        # out = self.fc(out)

        return out
        # return out,feature

# then we can define those model 
def m_resnet18(num_classes, **kwargs):
    model = m_ResNet(Basicblock,num_classes,[2,2,2,2],**kwargs)
    return model

def m_resnet34(num_classes, **kwargs):
    model = m_ResNet(Basicblock,num_classes,[3,4,6,3],**kwargs)
    return model

def m_resnet50(num_classes, **kwargs):
    model = m_ResNet(Bottleneck,num_classes,[3,4,6,3],**kwargs)
    return model

def m_resnet101(num_classes, **kwargs):
    model = m_ResNet(Bottleneck,num_classes,[3,4,23,3],**kwargs)
    return model

def m_resnet152(num_classes, **kwargs):
    model = m_ResNet(Bottleneck,num_classes,[3,8,36,3],**kwargs)
    return model

class chooseResNet():
    def __init__(self,num_classes,type='resnet50'):
        super(chooseResNet,self).__init__()
        self.num_classes = num_classes
        self.type = type 
    
    def GetModel(self):
        if self.type == 'resnet18':
            model = m_resnet18(self.num_classes)
        elif self.type == 'resnet34':
            model = m_resnet34(self.num_classes)
        elif self.type == 'resnet50':
            model = m_resnet50(self.num_classes)
        elif self.type == 'resnet101':
            model = m_resnet101(self.num_classes)
        elif self.type == 'resnet152':
            model = m_resnet152(self.num_classes)
        else:
            raise ValueError('not such model')
        
        return model

# Test you model design here.
if __name__ == "__main__":
    modela = m_resnet50(100)
    
    tempdata = torch.randn(32,3,56,56)
    # out = modela(tempdata)
    # stride and padding part will be the problem
    # 通过输入进行调试看看情况。
    # stat(modela,(3,224,224))
    summary(modela.cuda(),input_size = (3,56,56),batch_size=1)
    

