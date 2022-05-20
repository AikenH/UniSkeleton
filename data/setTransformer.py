"""
@ AikenHong 2021
@ Desc:
    using this file to manage those transforms may append in out experiment
    regiest it to activate the config setting. then we can easily got what we want.
@ 
"""

from torchvision.transforms import transforms
from data.dataAugment import myAugmentation as ma, GaussianBlur

# setup some transforms combinations.
cifar10_transformer_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # transforms.Resize(224),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.5), #ef
                transforms.RandomCrop(32,padding=4),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465],
                [0.2023, 0.1994, 0.2010]),
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 2), value=0, inplace=False) #ef
            ])
cifar10_transformer_val = transforms.Compose([
                # transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465],
                [0.2023, 0.1994, 0.2010])
            ])

cifar100_transformer_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32,padding=4),
                # transforms.Resize(22  4),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.5), #ef
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 2), value=0, inplace=False) #ef
            ])
cifar100_augs_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32,padding=4),
                # transforms.Resize(224),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.5), #ef
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 2), value=0, inplace=False) #ef
            ])
cifar100_transformer_val = transforms.Compose([
                # transforms.Resize(224),
                transforms.ToTensor(), 
                transforms.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
            ])

cifar100PIL_transformer_train = transforms.Compose([
                transforms.ToPILImage(mode='RGB'),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32,padding=4),
                # transforms.Resize(22  4),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.5), #ef
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 2), value=0, inplace=False) #ef
            ])

color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
simclr_transformer = transforms.Compose([
                transforms.RandomResizedCrop(size=32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
                ])

simclr_transformer_10 = transforms.Compose([
                transforms.RandomResizedCrop(size=32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465],
                [0.2023, 0.1994, 0.2010]),
                ])

def select_transform(transtype,istrain,**kwargs):
    """
    Input:
        transtype, istrain
    Return:
        transform
    """
    if transtype == 'cifar100':
        transformer = cifar100_transformer_train if istrain else cifar100_transformer_val
    
    elif transtype == 'cifar10':
        transformer = cifar10_transformer_train if istrain else cifar10_transformer_val

    elif transtype == 'cifar100+':
        transformer = cifar100_augs_train if istrain else cifar100_transformer_val

    elif transtype == 'cifar100pil':
        transformer = cifar100PIL_transformer_train if istrain else cifar100_transformer_val
    
    elif transtype == 'simclr':
        transformer = ma.simclr_trainsformer(**kwargs) if istrain else None

    elif transtype == 'base':
        transformer = ma.base_transformer(**kwargs) if istrain else None
    else:
        transformer = None
    return transformer
