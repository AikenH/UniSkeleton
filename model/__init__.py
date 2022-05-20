"""@Aiken 2021 model design
# consider whether to integrate feature model and classifer
# how to integrate two model and train in one process
"""
from torch import nn
import torchvision.models as t_models
from layers.activations import activation_loader
# from model.bak_swin import swin_t

def M_select(modelName, num_cls=None, pretrain=False, in_dim=None, hidden_layers=None, dropout=0,
            activation=None, ln=False, *args,**kwargs):
    """using this method to get the right Module including classifier and backbone""" 
    if 'resnet' in modelName:
        from model.ResNet import chooseResNet
        resnet = chooseResNet(num_cls,type = modelName)
        model = resnet.GetModel()

    elif 'EfficientNet' in modelName:
        from model.EfficientNet import ChooseEfficientNet
        efficientnet = ChooseEfficientNet(num_cls,type= modelName)
        model = efficientnet.GetModel()

    elif 'Swin' in modelName:
        from model.ViTs import SelectTrans
        transformer = SelectTrans(num_cls,type = modelName)
        model = transformer.GetModel()

        # model = swin_t()
    elif 'cifar_rs' in modelName:
        from model.small_ResNet import cifarResNet 
        resnet = cifarResNet(num_cls,type = modelName)
        model = resnet.GetModel()
    
    elif 'res32' in modelName:
        # i found this model in the bbn, and we use the optionA for the cifar
        # from model.ResNet32 import res32_cifar
        from model.ResNet32bbn import res32_cifar
        model = res32_cifar()

    elif 'mlp' in modelName:
        from layers.classifier import m_MLP
        assert in_dim is not None, "in_dim must be given"
        activation = activation_loader(activation)
        model = m_MLP(input_size=in_dim, num_classes=num_cls, hidden_layers=hidden_layers, 
                    dropout=dropout, activation=activation, ln=ln)

    elif 'causal' in modelName:
        from layers.CausalNormClassifier import Causal_Norm_Classifier
        model = Causal_Norm_Classifier(num_cls,in_dim,**kwargs)
    
    elif 'disalign' in modelName:
        from layers.classifier import DisAlignLinear
        model = DisAlignLinear(num_cls=num_cls,in_dim=in_dim,**kwargs)
    
    else:
        raise NotImplementedError('NOT IMPLEMENTED, add it in ./model/__init__')

    return model

class Assembler(nn.Module):
    """using this class to assembly model"""
    def __init__(self, feat_model, cls_model, num_cls, pretrain=False, in_dim=None, hidden_layers=None,
                feature_vis=False, dropout=0, activation=None, ln=False):
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
                                        dropout=dropout, activation=activation, ln=ln)

        # setting the input style 
        self.feature_vis = feature_vis
        
    def forward(self, x, **kwargs):
        feature = self.backbone(x)
        if self.classifier is not None:
            res = self.classifier(feature,**kwargs)
        else:
            res = feature
            self.feature_vis = False
        
        if self.feature_vis:
            return feature, res
        else:
            return res

    def _expand_dim(self, num_cls, re_init=True, hidden_layers=None, *args, **kwargs):
        """using a new classifier"""
        self.classifier._expand_dim(num_cls, re_init, hidden_layers, *args, **kwargs)
        return None
    
    def _get_projector(self, projector):
        """
        passing a projector(after train) to the classifier
        """
        self.projector = projector
        return None
    
    def _freeze(self, type='backbone'):
        """freeze the backbone or classifier"""
        if type == 'backbone':
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif type == 'classifier':
            for param in self.classifier.parameters():
                param.requires_grad = False
        else:
            raise ValueError('type should be backbone or classifier')
        return None