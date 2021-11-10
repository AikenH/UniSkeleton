"""@Aiken 2021 model design
# consider whether to integrate feature model and classifer
# how to integrate two model and train in one process
"""
# if necessary, we can use try inport to import the model by a Hard-Code Dict
import torchvision.models as t_models
from torch import nn
from model.ResNet import chooseResNet
from model.small_ResNet import cifarResNet 
from model.EfficientNet import ChooseEfficientNet
from model.ViTs import SelectTrans
from layers.classifier import m_MLP
from layers.activations import activation_loader
# from model.bak_swin import swin_t

def M_select(model_t, num_class=None, pretrain=False, in_dim=None, hidden_layers=None, dropout=0,
            activation=None, *args,**kwargs):
    """using this method to get the right Module including classifier and backbone""" 
    if 'resnet' in model_t:
        resnet = chooseResNet(num_class,type = model_t)
        model = resnet.GetModel()

    elif 'EfficientNet' in model_t:
        efficientnet = ChooseEfficientNet(num_class,type= model_t)
        model = efficientnet.GetModel()

    elif 'Swin' in model_t:
        transformer = SelectTrans(num_class,type = model_t)
        model = transformer.GetModel()

        # model = swin_t()
    elif 'cifar_rs' in model_t:
        resnet = cifarResNet(num_class,type = model_t)
        model = resnet.GetModel()

    elif 'mlp' in model_t:
        assert in_dim is not None, "in_dim must be given"
        activation = activation_loader(activation)
        model = m_MLP(input_size=in_dim, num_classes=num_class, hidden_layers=hidden_layers, 
                    dropout=dropout, activation=activation)
    
    else:
        raise NotImplementedError('NOT IMPLEMENTED, add it in ./model/__init__')

    return model

class Assembler(nn.Module):
    """using this class to assembly model"""
    def __init__(self, feat_model, cls_model, num_cls, pretrain=False, in_dim=None, hidden_layers=None,
                feature_vis=False, dropout=0, activation=None):
        super(Assembler, self).__init__()
        # init the backbone and the classifier
        self.backbone = M_select(feat_model, num_cls, pretrain)
        self.classifier = None

        # get the classifier by the out_dim of backbone
        # ATTENTION: we need give the in_dim or difine it in the model
        if cls_model != 'Defaults':
            if in_dim is None:
                try:
                    in_dim = self.backbone.out_dim
                except:
                    raise ValueError('backbone has no out_dim, please check or add in model and yaml')

            self.classifier = M_select(cls_model, num_cls, in_dim = in_dim, hidden_layers=hidden_layers, 
                                        dropout=dropout, activation=activation)

        # setting the input style 
        self.feature_vis = feature_vis
        
    def forward(self, x):
        feature = self.backbone(x)
        if self.classifier is not None:
            res = self.classifier(feature)
        else:
            res = feature
            self.feature_vis = False
        
        if self.feature_vis:
            return feature, res
        else:
            return res
