"""
Using this doc to manage those data augmentation and the transformer setting.
I need to be more familiar with the transformer and the collate_fn and how to combine it.

- when we using those download dataset(like cifar100), we better use transform in torchvision
- others, using the trasformer with albumentations may be better and flexible.

@Article{info11020125,
    AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
    TITLE = {Albumentations: Fast and Flexible Image Augmentations},
    JOURNAL = {Information},
    VOLUME = {11},
    YEAR = {2020},
    NUMBER = {2},
    ARTICLE-NUMBER = {125},
    URL = {https://www.mdpi.com/2078-2489/11/2/125},
    ISSN = {2078-2489},
    DOI = {10.3390/info11020125}
}

@Author: AikenHone 2021
@Email: h.aiken.970@gmail.com
@Purpose: 
    using static class methods to write data augmentation functions,
    and then use them in the main script, call those in __getitem__
"""
import torch
from torch import nn
from torchvision.transforms import transforms
import cv2
import numpy as np
import albumentations as Augs

class myAugmentation():
    """
    Principle:
        sometimes we want a augmentation which define those params in advance
        maybe in such a situation, instaniated a class is better way.
    Default:
        Do not create instance for this class, we call the classmethods,
        Define class params in the beginning of the class.
    """
    
    def __init__(self):
        """
        placeholder
        """
        print('We Do not want this class to be instantiated, it wont bring any benifit')
    
    @classmethod
    def base_transformer(cls,i_size):
        """this transformer is used to replace operation on PIL(torchvision)"""
        transformer = Augs.Compose([
            Augs.HorizontalFlip(p=0.5),
            Augs.RandomCrop(height=i_size, width=i_size, p=0.5),
            Augs.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            Augs.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])
        return transformer

    @classmethod
    def better_transformer(cls, mean, std, i_size):
        # this is a better transformer, it can be used to replace operation on PIL(torchvision)
        transformer = Augs.Compose([
            Augs.HorizontalFlip(p=0.5),
            Augs.RandomCrop(height=i_size, width=i_size, p=0.5),
            Augs.RandomBrightnessContrast(),
            Augs.Normalize(mean,std),
        ])
        return transformer
    
    @classmethod
    def simclr_trainsformer(cls, i_size):
        transformer = Augs.Compose([
            Augs.HorizontalFlip(p=0.5),
            Augs.RandomResizedCrop(i_size,i_size),
            Augs.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            Augs.GaussianBlur((3,5)),
        ])
        return transformer

    @classmethod
    def list_of_transformers(cls, i_size, *args, **kwargs):
        """using this method to generate many augmentation version of image"""
        transform = [
            Augs.HorizontalFlip(p=1),
            Augs.RandomCrop(height=i_size, width=i_size, p=1),
            Augs.CenterCrop(height=i_size, width=i_size, p=1),
            Augs.RandomBrightnessContrast(p=1),
            Augs.Resize(i_size,i_size),
        ]
        return transform

    @classmethod
    def mix_up(cls, datas, targets, alpha=1):
        """
        @reference: mixup: https://arxiv.org/pdf/1710.09412.pdf
        @source code: https://github.com/facebookresearch/mixup-cifar10/
        This augmentation need be used with a specific Loss Function
        Which will be registered in the loss-selector function using mix_up.

        args:
            datas: list of images
            targets: list of labels
            alpha: float, the weight of the mixup augmentation

        return:
            mixed_datas: list of mixed images
            targets_bf: the targets before shuffled
            targets_af: the targets after shuffled
            lam: the weight of the mixup augmentation
        """
        # get the basic paramerters
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        batch_size = datas.size(0)

        # shuffle the index and transfer to the cuda 
        if torch.cuda.is_available():
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        
        # mixup the data and the labels by the beta distribution
        mixed_datas = lam * datas + (1 - lam) * datas[index, :]
        targets_bf, targets_af = targets, targets[index]
        
        return mixed_datas, targets_bf, targets_af, lam

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

class ContrastiveLearningViewGenerator():
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

class ContrastiveLearningViewGenerator_v2():
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(image=x)['image'] for i in range(self.n_views)]

if __name__ == "__main__":
    # test the class methods
    # read and visualize the origin Img
    save_path = '../DownloadData/ImageNet/train/n02102040/n02102040_986.JPEG'
    img = cv2.imread(save_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('lena.png',img)
    # cv2.imshow("origin_img", img)
    # cv2.waitKey(0) # manually close the window

    transformer = myAugmentation.list_of_transformers(224)
    img_transformed = transformer[0](image=img)['image']
    cv2.imwrite('trans.png',img_transformed)
    # cv2.imshow("img_transformed", img_transformed)
    # cv2.waitKey(0) # manually close the window