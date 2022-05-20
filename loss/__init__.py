"""@AikenH 2021 loss selection design part
LOSS FUNCTION SELECTOR
regiester your loss function here
"""
import torch.nn as nn

def L_select(LOSSTYPE,*args,**kwargs):
    """
    if we want define a loss using config.yaml,
    Then we need to register the loss function here, custom ur key word
    """
    if LOSSTYPE.lower() == 'mse':
        return nn.MSELoss()

    elif LOSSTYPE.lower() == 'l1':
        return nn.L1Loss()

    elif LOSSTYPE.lower() == 'huber':
        return nn.SmoothL1Loss()

    elif LOSSTYPE.lower() == 'bce':
        return nn.BCELoss()

    elif LOSSTYPE.lower() == 'ce':
        return nn.CrossEntropyLoss()

    elif LOSSTYPE.lower() == 'nll':
        return nn.NLLLoss()

    elif LOSSTYPE.lower() == 'kl':
        from loss.custom_loss import kl_loss
        return kl_loss

    elif LOSSTYPE.lower() == 'mix_up':
        # [ ]: how to combine other loss in the config, we need to rethink it 
        from loss.custom_loss import mixup_criterion
        return mixup_criterion

    elif LOSSTYPE.lower() == 'kd':
        from loss.KD_loss import kdloss
        return kdloss
    
    elif LOSSTYPE.lower() == 'kd_c':
        from loss.KD_loss import KD
        kdloss = KD(**kwargs)
        return kdloss
    
    elif LOSSTYPE.lower() == 'kd_smooth':
        from loss.KD_loss import LabelSmoothingKD
        kdloss = LabelSmoothingKD(**kwargs)
        return kdloss
    
    elif LOSSTYPE.lower() == 'icarl':
        from loss.KD_loss import iCaRL_loss
        return iCaRL_loss
    
    elif LOSSTYPE.lower() == 'icarl_c':
        from loss.KD_loss import ICARL
        iCaRL_loss = ICARL(**kwargs)
        return iCaRL_loss
    
    elif LOSSTYPE.lower() == 'arcface':
        from loss.angular_loss import AngularPenaltySMLoss
        arcface = AngularPenaltySMLoss(loss_type='arcface')
        return arcface

    elif LOSSTYPE.lower() == 'ntxent':
        from loss.NTXent_loss import NTXentLoss
        # we need to set up this params according to the GPU devices.
        # https://docs.lightly.ai/tutorials/package/tutorial_moco_memory_bank.html
        # batch_size = 512 will take around 5G of GPU, 
        # we can set the memory_size based on this
        memory_size = kwargs.get('memory_size', 0)
        temperature = kwargs.get('temperature', 0.07)
        return NTXentLoss(temperature=temperature, memory_bank_size=memory_size)
    
    elif LOSSTYPE.lower() == 'supcontrast':
        from loss.supContrast_loss import SupConLoss
        temperature = kwargs.get('temperature', 0.07)
        return SupConLoss(temperature=temperature)
    
    elif LOSSTYPE.lower() == 'disalign':
        from loss.longtailed_loss import DisAlignLoss
        if not kwargs.get('cls_num_list', None):
            raise ValueError('the num of each classes should be pass in, \
                            we using this to align the bias')
        return DisAlignLoss(**kwargs)

    else:
        raise NotImplementedError('NOT IMPLEMENTED, add it in L_select')
