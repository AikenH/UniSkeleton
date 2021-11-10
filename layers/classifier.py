"""
@AikenH 2021  classifier layer
# design specific classifer here to integrate model
# which is used to replace the normal fc layer
"""
import torch
from torch import nn 
from collections import OrderedDict

from torchstat import stat 
from torchsummary import summary

class m_MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layers:list=None, dropout=0,
                activation=nn.ReLU(inplace= True),):
        super(m_MLP, self).__init__()
        # # basic parameters
        # self.input_size = input_size
        # self.num_classes = num_classes
        # self.hidden_layers = hidden_layers

        # layers
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.fc = self._make_layer(input_size, num_classes, hidden_layers)
            
    def forward(self,x):
        res = self.avgpool(x)
        res = res.view(res.size(0), -1)
        # res = self.dropout(res)
        res = self.fc(res)
        return res 
    
    def _make_layer(self,input_size, num_classes, hidden_layers=None):
        """create the main body of the mlp layer"""
        if hidden_layers is None:
            layers = [nn.Linear(input_size, num_classes)]
            if self.activation is not None:
                layers.append(self.activation)
        else:
            layers = []
            for i in range(len(hidden_layers)):
                if i == 0:
                    layers.append(nn.Linear(input_size, hidden_layers[i]))
                else:
                    layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                if self.activation is not None:
                    layers.append(self.activation)
            layers.append(nn.Linear(hidden_layers[-1], num_classes))
        return nn.Sequential(*layers)

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


# ------------------------- Test --------------------------
if __name__ == "__main__":

    tempdata = torch.randn(32,3,224,224)
    model = ...

    # choose one method to visualize the parameters
    # stat(model,(3,224,224))
    summary(model.cuda(),input_size=(3,224,224),batch_size=1)
