"""
@AikenH 2021  classifier layer
# design specific classifer here to integrate model
# which is used to replace the normal fc layer
"""
import copy
import torch
from torch import nn 
import torch.nn.functional as F
from collections import OrderedDict

from torchstat import stat 
from torchsummary import summary

# ATTENTION: from this method we can learn how to design a basic unit for a neural network
class NormedLinear(nn.Module):
    def __init__(self,feat_dim, num_classes):
        super().__init__()
        self.weight = nn.parameter(torch.Tensor(feat_dim, num_classes))
        self.weight.data.uniform(-1,1).renorm_(2,1,1e-5).mul_(1e5)

    def forward(self,x):
        return F.normalize(x,dim=1).mm(F.normalize(self.weight, dim=0))

class DisAlignLinear(nn.Module):
    def __init__(self, in_dim, num_cls, use_norm=True):
        super().__init__()
        self.classifier = NormedLinear(in_dim, num_cls) if use_norm else nn.Linear(in_dim, num_cls)
        self.learned_magnitude = nn.Parameter(torch.ones(1, num_cls))
        self.learned_margin = nn.Parameter(torch.zeros(1, num_cls))
        self.confidence_layer = nn.Linear(in_dim, 1)
        torch.nn.init.constant_(self.confidence_layer.weight, 0.1)

    def forward(self, x):
        output = self.classifier(x)
        confidence = self.confidence_layer(x).sigmoid()
        return (1 + confidence * self.learned_magnitude) * output + confidence * self.learned_margin

class m_MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layers:list=None, dropout=None,
                activation=nn.ReLU(inplace=True), ln=False):
        super(m_MLP, self).__init__()
        # # basic parameters
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers

        # layers
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.ln = ln
        self.fc = self._make_layer(input_size, num_classes, hidden_layers)
       
    def forward(self,x,**kwargs):
        # we need to divide the input from the labelge
        if x.dim() >=3 : res = self.avgpool(x)
        else: res = x

        res = res.view(res.size(0), -1)
        if self.dropout is not None:
            res = self.dropout(res)
        # res = self.dropout(res)
        res = self.fc(res)
        return res 
    
    def _make_layer(self,input_size, num_classes, hidden_layers=None):
        """create the main body of the mlp layer"""
        if hidden_layers is None:
            layers = []
            
            if self.ln:
                layers.append(nn.LayerNorm(input_size))
            
            layers.append(nn.Linear(input_size, num_classes))

            if self.activation is not None:
                layers.append(self.activation)
        else:
            layers = []
            for i in range(len(hidden_layers)):
                if i == 0:
                    layers.append(nn.Linear(input_size, hidden_layers[i],bias=True))
                else:
                    layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                
                if self.activation is not None:
                    layers.append(self.activation)
            layers.append(nn.Linear(hidden_layers[-1], num_classes))
        return nn.Sequential(*layers)

    def _expand_dim(self, num_cls, re_init=True, hidden_layers=None, *args, **kwargs):
        """
        republish the classifier for the distill learning
        using this method to duplicate the classifier easier
        """
        # save those parameters in advance.
        weights_list, bias_list = [], []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                weights_list.append(copy.deepcopy(m.weight.data))
                bias_list.append(copy.deepcopy(m.bias.data))

        # build a new fc layer
        self.fc = self._make_layer(self.input_size, num_cls, hidden_layers)
        
        # re-init the rest of model weights and bias
        if re_init is not None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight.data)
                    m.bias.data.zero_()
        
        # passing the old parameters to the new classifier for distill
        if kwargs.get('distill', False):
            index = 0
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    while index == len(weights_list)-1 :
                        m.weight.data[:self.num_classes] = weights_list[index-1]
                        m.bias.data[:self.num_classes] = bias_list[index-1]
                        break
                    index +=1
            # check the lens of the list and the parameters
            assert index == len(weights_list)

        return None

class evidential_layer(nn.Module):
    def __init__(self,inCha,outCha,*args,**kwargs):
        super().__init__()
        """ 这个全连接的输入会代替原本的全连接，然后最后是将好几个concate起来的"""
        # 如果使用evidential layer，输出的格式和长度会发生改变，具体使用的时候要知道。
        
        # 拓展全连接层
        self.fc = nn.Linear(inCha,outCha)
        self.probability = nn.Linear(inCha,3)
        self.evidence = nn.Softplus()
    
    def forward(self,x):
        out = self.fc(x)
        mu,logv,logalpha,logbeta = torch.split(out, 4, dim=-1)
        
        v = self.evidence(logv)
        alpha = self.evidence(logalpha)
        beta = self.evidence(logbeta)

        return torch.cat([mu,v,alpha,beta],dim=-1)

if __name__ == "__main__":

    tempdata = torch.randn(32,3,224,224)
    model = ...

    # choose one method to visualize the parameters
    # stat(model,(3,224,224))
    summary(model.cuda(),input_size=(3,224,224),batch_size=1)
