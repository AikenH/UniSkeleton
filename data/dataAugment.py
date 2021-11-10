""" 
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

@Authod: AikenHone 2021
@Email: h.aiken.970@gmail.com
@Purpose: 
    using static class methods to write data augmentation functions,
    and then use them in the main script, call those in __getitem__
"""
import numpy as np

# package for image augmentation
import cv2
import albumentations as Augs

class myAugmentation():
    """
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
            Augs.RandomBrightnessContrast(),
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
    def list_of_transformers(cls, i_size, *args, **kwargs):
        """using this method to generate many augmentation version of image"""
        transforms = [
            Augs.HorizontalFlip(p=1),
            Augs.RandomCrop(height=i_size, width=i_size, p=1),
            Augs.CenterCrop(height=i_size, width=i_size, p=1),
            Augs.RandomBrightnessContrast(p=1),
            Augs.Resize(i_size,i_size),
        ]
        return transforms
    
    @classmethod
    def open_mix():
        """
        
        """

        pass

    @classmethod
    def mix_up():
        """
        mixup: https://arxiv.org/pdf/1710.09412.pdf
        """

        pass

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