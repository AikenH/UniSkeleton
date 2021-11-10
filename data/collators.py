"""
@Author: AikenHong
@Date: 2021.11.05
@Email: h.aiken.970@gmail.com
@Description: writing diff collate_fn in the file
                1. solve the diff size of img in one batch if we need 
                2. Doing diff augs for one img in batch as positive label
                3. we can make it tensor or GPU in here
                4. TBD
@qidong↓
if we just write single method, we must use Lambda function to pass params
like:
    loader = DataLoader(collate_fn=lambda x: collate_fn(x, **params))
or we can just define a callable class, pass params using the initial progress
"""
# basic module
import torch
import numpy as np
from torchvision import transforms

class collate():
    def __init__(self, to_tensor=False, i_size=None, transforms=None): 
        # add all the params we need here
        self.i_size = i_size
        self.augs = transforms
        self.to_tensor = to_tensor

    def __call__(self, batch):
        # data is a list of img and label
        labels = [data[1] for data in batch]
        imgs = []
        for img in batch:
            if self.i_size is not None:
                img = self._resize(img)
            
            if self.augs is not None:
                img = self._augs(img)

            # this part must after aug which using np.array 
            if self.to_tensor:
                img = self._totensor(img)
            imgs.append(img)
        
        # make it batches
        labels = torch.stack(labels)
        imgs = torch.stack(imgs)
        return [imgs, labels]

    def _resize(self, img):
        # resize img
        return transforms.Resize(self.i_size)(img)

    def _augs(self, img):
        # augmentation
        res = []
        for transform in self.augs:
            # if img is tensor, we must use img.numpy(), but we better not 
            if type(img) == torch.Tensor:
                img = img.numpy()
                self.to_tensor = True

            img = transform(image=img)['image']
            res.append(img)
        return res

    def _totensor(self, img):
        # convert img to tensor
        return transforms.ToTensor()(img)
    