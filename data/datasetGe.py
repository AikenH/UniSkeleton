"""
# @AikenHong 2021
# Purpose: Simulate real data environment scenarios 
# Method: Using Sampling to get Long Tailed of data 
#         Using FSL Sampling to get the New Classes data
#         Do basic transform in the __getitem__ and do not set default for it.
we should pass diff transformer for diff stage of training(val)
pass data augmentation to __getitem__ using TRANSFORM
#       if we want to dublicate a img or do more augmentation, we call func
        in training or in the collate_fn
"""
# python module
import os 
import glob
import json
import copy
# data reading module
import cv2
import numpy as np
from PIL import Image
# torch module
import torch
import torchvision
from torch.utils import data
from torchvision.transforms import transforms
# my script
from util.utils import mapping_order_list
from data.dataAugment import myAugmentation
from data.setTransformer import select_transform

DOWNLOAD_DATA = ['cifar100','cifar10','MNIST','FashionMNIST']
LOCAL_DATA = ['ImageNet','MiniImageNet', 'LocalFiles']

def sort_dataset(dataset):
    # !! 同步排序有很多更好的方法，这里可以借鉴一下，进行更新
    # the num_new_class can be calculate by some formula, but in this part make it HARD
    # sort those data and label which make it easier to del class.
    num_data = len(dataset)
    up_limit = pow(10, len(str(num_data)))
    index = [index /up_limit for index in range(num_data)]
    
    # using this mark to sort the data
    for i, _ in enumerate(dataset.targets):
        dataset.targets[i] += index[i]
    dataset.targets.sort()
    
    # get the new order 
    new_order = [int((target - int(target)) * up_limit) for target in dataset.targets] 
    dataset.targets = [int(target) for target in dataset.targets]
    # it's necessary for us to swith to list or not?
    dataset.data = list(np.array(dataset.data)[new_order,...])

    return None

def download_data(datatype='cifar100', istrain=0, save_path=None, transform=None, *args, **kwargs):
    """
    Input:
        datatype, istrain, save_path
    Return:
        origin dataset which is not extract new class
    """
    istrain = True if not istrain else False
    # registegy default path for linux,and ignore windows for now
    save_path = r'/workspace/DownloadData' if save_path is None else save_path
    # specific transform in each type of data,
    if transform is None:
        transform = select_transform(datatype,istrain)
    if datatype == 'cifar100':
        # the origin Image is 32*32 if without padding the crop is useless
        dataset = torchvision.datasets.CIFAR100(save_path, train=istrain,
                            transform=transform, download= True)
    elif datatype == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(save_path, train=istrain,
                            transform=transform, download= True)
    elif datatype == 'MNIST':
        dataset = torchvision.datasets.MNIST(save_path, train=istrain,
                            transform=transform, download=True)
    elif datatype == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST(save_path, train=istrain,
                            transform =transform, download= True)
    else:
        raise NotImplementedError('Need to Add this dataset in the GetData.py')

    return dataset

def extract_new_class(dataset, num_new_classes, num_class, sort=False, random_seed=None, order_list=None,*args, **kwargs):
    """
    dataset : 
        deliver the origin dataset to this func.
    num_new_classes : 
        how much new classes we want to generate.
        others will be consideredas konwn classes which can reach full data of it.
        In contrast the new class uses Few-Shot strategy for data extraction
    *args: 
        'sort' which will enable the sort part for those dataset is shuffled like cifar
    return: 
        return Known dataset and New Classes dataset
    """
    # the num_new_class can be calculate by some formula, but in this part make it HARD
    if sort:
        sort_dataset(dataset)
        
    # set the random seed to get the same sampler of class
    if random_seed is not None:
        np.random.seed(random_seed)

    # random define the new_class
    new_order_dict = list(range(num_class))
    if kwargs.get('shuffle', True):
        np.random.shuffle(new_order_dict)
    
    if order_list is not None:
        new_order_dict = order_list

    new_classes = new_order_dict[num_class-num_new_classes:]
    
    if kwargs.get('fix', False):
        # NOTE: store those hyper-class for confi-test
        # https://cloud.tencent.com/developer/article/1679335
        new_classes = [3, 7, 13, 27, 41, 42, 43, 48, 52, 56, 57, 59, 69, 80, 81, 88, 89, 90, 96, 97]
        new_order_dict = [i for i in range(num_class) if i not in new_classes]
        new_order_dict += new_classes
    
    # enable this for the test process
    if num_new_classes == 0: new_classes = []

    new_index = [target in new_classes for target in dataset.targets]
    
    # Dublicate the basic dataset
    # using diff index to generate those datasets.
    new_class_dataset = copy.deepcopy(dataset)

    new_data, new_targets = [], []
    old_data, old_targets = [], []
    for i in range(len(dataset)):
        if new_index[i] == True: 
            new_data.append(dataset.data[i])
            new_targets.append(dataset.targets[i])
        else:
            old_data.append(dataset.data[i])
            old_targets.append(dataset.targets[i])

    # assign those list to the dataset
    remap_targets = mapping_order_list(new_order_dict,old_targets)
    dataset.data, dataset.targets = old_data, remap_targets

    remap_new_targets = mapping_order_list(new_order_dict,new_targets)
    new_class_dataset.data, new_class_dataset.targets = new_data, remap_new_targets

    return dataset, new_class_dataset

def generate_unbalance_class(dataset, num_cls, factor=0.3, strategy='step', decay=0.1, random_seed=None, *args, **kwargs):
    """
    DATASET: 
        deliver the origin dataset to this func.
    FACTOR:
        using this factor to make unbalance
    STRATEGY:
        which stractegy we use (step,exp) to make unbalance
    RETURN:
        unbalance dataset.(Dublicate or not and Modify from origin)
    """
    # calculate the sample rate of samples corresponding to each class
    ratio_per_cls = []
    if strategy == 'step':
        ratio_per_cls += [1] * (num_cls//2)
        # ratio = decay
        for cls_idx in range(1,num_cls//2 +1):
            # if cls_idx % (num_cls // 10) == 0: 
                # decay += 0.1
                # ratio = ratio * decay
            ratio_per_cls.append(factor)

    elif strategy == 'exp':
        for cls_idx in range(num_cls):
            ratio = factor ** (cls_idx/(num_cls-1))
            ratio_per_cls.append(ratio)
    else:
        ratio_per_cls = [1] * len(dataset)
    
    # using the sampling rate to get data
    targets_bak = np.array(dataset.targets)
    classes = np.unique(targets_bak)
    np.random.seed(random_seed)
    np.random.shuffle(classes)
    # the final data
    imb_data,imb_targets = [], []

    # this part will make the list ordered, we need to shuffle this for the future use
    for label, ratio in zip(classes, ratio_per_cls):
        # find all this class
        idx = np.where(targets_bak == label)[0]
        np.random.shuffle(idx)
        num_pre_cls = int(len(idx) * ratio)
        select_idx = idx[:num_pre_cls]
        imb_data += [dataset.data[i] for i in select_idx]
        imb_targets += [label] * num_pre_cls
    
    # # [ ]: improve the shuffle part in there
    # np.random.seed(888)
    # np.random.shuffle(imb_data)
    # np.random.seed(888)
    # np.random.shuffle(imb_targets)

    dataset.data, dataset.targets = imb_data, imb_targets
    if 'fix_order' in args:
        return [dataset, classes]
    return dataset

class AllData(data.Dataset):
    """
    Init: save_path, train, datatype
    Return: origin dataset for Local data
    """

    def __init__(self, datatype='ImageNet', istrain=0, save_path=r'/workspace/DownloadData/ImageNet',
                transform=None, needMap=False, datalist=None, targetlist=None, *args, **kwargs):
        super(AllData,self).__init__()
        # setting basic params
        self.isDownload = False
        self.maping = None
        self.transform = transform
        self.isNpy = False
        self.datatype = datatype
        self.out_idx = False
        self.train = istrain
        # diff process for diff dataset 
        if datatype in LOCAL_DATA:
            # Set the basic Params 
            self.SUBFIX_NEED = ['.jpg', '.png', '.JPEG','.npy']
            if needMap:
                self.maping = self._generate_mapping(save_path)
                assert self.maping is not None, "not generate the mapping correctly"
            
            # Get the final path for the data 
            self.train = istrain
            if(self.train == 0): save_path = os.path.join(save_path,'train')
            elif(self.train == 1): save_path = os.path.join(save_path, 'test')
            else: 
                # if there is not val data, we can split test or train to do that
                t_path = os.path.join(save_path,'val')
                if not os.path.exists(t_path): 
                    save_path = os.path.join(save_path, 'test')
                else:
                    save_path = t_path

            # get the origin dataset
            self.data, self.targets = self.get_data_dir(save_path)
        
        elif datatype in DOWNLOAD_DATA:
            self.isDownload = True
            if self.transform is not None:
                self.tmp_dataset = download_data(datatype, istrain, save_path, transform=self.transform)
            else:
                self.tmp_dataset = download_data(datatype, istrain, save_path)
            self.targets, self.data = self.tmp_dataset.targets, self.tmp_dataset.data
            self.transform = self.tmp_dataset.transform
        
        elif datatype == "Given":
            self.data, self.targets = datalist, targetlist
            
        else:
            raise NotImplementedError('this data type is not define, you need to add it in getdata.py')
        
        return None

    def __getitem__(self, index):
        # 这里需要考虑两种情况：第一种是读取进来就是字典的情况，我们默认为已经经过处理了，
        # 第二种情况是，读取进来是单张图片的情况，这种情况下，我们要调用函数来对对其进行处理
        if self.datatype == 'Given':
            image, label = self.data[index], self.targets[index]
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)

            if self.transform is not None:
                try:
                    image = self.transform(image=image)['image']
                    image = transforms.ToTensor()(image)

                except:
                    image = self.transform(image)
                
        elif self.isDownload == False:
            if not self.isNpy:
                # using cv2 to load jpg or sth. else
                image = cv2.imread(self.data[index])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # load npy is another method
                image = np.load(self.data[index], allow_pickle=True)
                image = image.item()
            label = self.targets[index]
            
            # this operation can be intergrate in the collate_fn in dataloader
            # which is use to avoid data shope is not the same
            # if image.mode != 'RGB':
            #     image = image.convert("RGB")
            if not isinstance(image, dict):
                if self.transform is not None:
                    image = self.transform(image)

        else:
            # image,label = self.tmp_dataset[index]
            image, label = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            image = Image.fromarray(image)

            if self.transform is not None:
                image = self.transform(image)
            
        # for the diff output dimension ,we need to fix the num of it 
        # if the image we load is dict, we think its already been processed, so we just output
        # if not isinstance(image, dict):
        #     if self.transform is not None:
        #         image = self.transform(image)

        if self.out_idx: return index, image, label
            
        return image, label

    def __len__(self):
        return len(self.data)

    # script to get data and generate the mapping files 
    def get_data_dir(self, save_path):
        """this method is for those dataset which use dirname as data label """
        dataset, labelset = [], []
        
        for root, dirs, files in os.walk(save_path, topdown=True):
            if len(files) == 0: continue
            
            # get the file's subfix
            subfix = [os.path.splitext(x)[1] for x in files]
            self.isNpy = subfix[0] in ['.npy']
            subfix = [subfix[i] in self.SUBFIX_NEED for i in range(len(subfix))]
            imgs = [files[i] for i in range(len(subfix)) if subfix[i] == True]
            if (len(imgs) == 0): continue

            # using unified / in windows and linux 
            label = root.split('/')[-1]
            
            # mapping to the machine label (all the dir is the same class)
            if self.maping is not None:
                label = self.maping[label]
            else:
                label = int(label)
            
            # get the path of the image
            # add data(path) and label to le output
            labelset += [label] * len(imgs)
            dataset += [os.path.join(root,img) for img in imgs]
            assert len(labelset) == len(dataset), "the len of data and label should be the same"

        return dataset, labelset
    
    def set_out_idx(self, enable=False): 
        """Using this flag to change the status of getitem"""
        self.out_idx = enable
        return None

    def _generate_mapping(self, save_path):
        """
        Maping label(str) to Continuous Int
        generate and store as json
        """
        mapping_file = glob.glob(os.path.join(save_path,'*.json'))
        
        if len(mapping_file) != 0: 
            return mapping_file[0]
        else:
            mapping_file = {}
            train_dir = os.path.join(save_path,'train')

            # can write loop to exclude file in here if needed            
            targets = os.listdir(train_dir)
            
            # np.random.shuffle(targets)
            
            # generate the mapping relationship 
            for i, target in enumerate(targets):
                mapping_file[target] = int(i)
            
            # save it as json file
            filename = 'maping2index.json'
            filename = os.path.join(save_path,filename)

            with open(filename,'w') as f:
                json.dump(mapping_file,f, ensure_ascii=False)
        
        return mapping_file
    
    def _save_origin_dataset(self, datasetname='cifar100-ow', savepath=r'/workspace/DownloadData', *args, **kwargs):
        # get the dir path dir + dataset-name + 'phase'
        savepath = os.path.join(savepath, datasetname)
        TAGS = ['train', 'val', 'test']
        savepath = os.path.join(savepath, TAGS[self.train])
        if not os.path.exists(savepath): os.makedirs(savepath)

        # load data from the list to get the origin data.
        # if using getitem will be after transform
        from tqdm.contrib import tzip
        for i, (img, label) in enumerate(tzip(self.data, self.targets)):
            cls_path = os.path.join(savepath, str(label))
            if not os.path.exists(cls_path): os.makedirs(cls_path)

            img_path = os.path.join(cls_path, str(i))
            np.save(img_path, img)

        return None

class DataSampler():
    def __init__(self, dataset, datatype='cifar100', istrain=0, 
                num_new=10,sort=False,imb_factor=0.3, strategy='step',decay=0.1,
                num_cls = None, random_seed=None, new_order=None, *args, **kwargs):
        # init the dataset attributes
        self.origin_dataset = dataset
        self.remain_dataset = None
        self.new_dataset = None
        self.imb_dataset = None
        self.dataype = datatype
        self.istrain = istrain

        # pass those params to this class
        self.num_new = num_new
        self.sort = sort
        self.factor = imb_factor
        self.strategy = strategy
        self.decay = decay
        self.random_seed = random_seed

        self.new_order = new_order

        # parsing num_relationship in here
        Name2Nums = {
            'cifar100': 100,
            'cifar10': 10,
            'MNIST': 10,
            'ImageNet': 1000
        }
        try:
            self.num_cls = Name2Nums[datatype] if not num_cls else num_cls
        except:
            raise ValueError("datatype is not in the list, you can add it in the datasampler or yaml")
        # set a flag for the imb_dataset
        self.imb_fi = False
        
    def _generate_new(self,num_new_classes,num_class,sort=False, random_seed=None, order_list=None, *args, **kwargs):
        dataset = copy.deepcopy(self.origin_dataset)
        if self.imb_dataset is not None:
            dataset = copy.deepcopy(self.imb_dataset)
        return extract_new_class(dataset,num_new_classes, num_class,sort, random_seed, order_list, *args, **kwargs)
        
    def _generate_imb(self,dataset, num_cls, factor, strategy='step',decay=0.1, random_seed=None, *args, **kwargs):
        tmp_dataset = copy.deepcopy(dataset)
        return generate_unbalance_class(tmp_dataset,num_cls, factor, strategy, decay, random_seed, *args, **kwargs)
    
    def get_remain(self, **kwargs): 
        # only excute this function the remain dataset is not defined
        if self.remain_dataset is None:
            self.remain_dataset, self.new_dataset = \
                self._generate_new(self.num_new,self.num_cls,self.sort, self.random_seed,order_list=self.new_order,**kwargs)
            self.num_cls -= self.num_new

        return self.remain_dataset
    
    def get_new(self, **kwargs): 
        # only excute this function the new dataset is not defined
        if self.new_dataset is None:
            self.remain_dataset, self.new_dataset = \
                self._generate_new(self.num_new,self.num_cls,self.sort,self.random_seed, order_list=self.new_order,**kwargs)
            self.num_cls -= self.num_new
            
        return self.new_dataset

    def get_imb(self,*args, **kwargs):
        # only excute this function if imb_dataset is not generate 
        if not self.imb_fi:
            dataset = self.origin_dataset if self.remain_dataset is None else self.remain_dataset
            self.imb_dataset = \
                self._generate_imb(dataset ,self.num_cls,self.factor,self.strategy,self.decay, self.random_seed,*args, **kwargs)
            if 'fix_order' in args:
                self.imb_dataset, self.new_order = self.imb_dataset
            self.imb_fi = True
        
        return self.imb_dataset
    
    def process(self):
        # which we use this function for both
        self.get_imb('fix_order')
        self.get_remain()

        return self.new_order

    def save_dataset(self, dataset, save_path, datatype, *args,**kwargs):
        """
        which can help us to reduce the preprocess cost
        1. we can save the dataset like ImageNet, make the dirname is the label
        2. using __getitems__ to save diff data augmentation version for the data
        """
        # generate the path of dir + set_name + 'train'
        if save_path is None: save_path = r'/workspace/DownloadData'
        save_path = os.path.join(save_path, datatype)
        
        TAGS = ['train', 'test', 'val']
        save_path = os.path.join(save_path, TAGS[self.istrain])

        # if the dir is not exist, create it
        if not os.path.exists(save_path): os.makedirs(save_path)
        
        # load datas by getitem and save it as npy files
        for i, (img, label) in enumerate(dataset):
            cls_path = os.path.join(save_path, str(label))
            if not os.path.exists(cls_path): os.makedirs(cls_path)
            
            # we need to identify the type of the data: dict or array
            # when we load it.
            if kwargs.get('aug') == True:
                temp = self.dublicate_augmentation(img,)
                img = temp
            
            img_path = os.path.join(cls_path, str(i))
            np.save(img_path, img)

        return None
    
    def dublicate_augmentation(self, img, strategy='all', random=False, i_size=224):
        """using diff strategy to dublicate the img, add it in save_dataset """
        # the default strategy, using this to propocess our data
        if strategy == 'all': 
            transforms = myAugmentation.list_of_transformers(i_size)
        elif strategy == 'base':
            transforms = myAugmentation.better_transformer(i_size)
        else:
            raise NotImplementedError("add it for yourself")
        
        # the main body of the augmentation
        imgs = [img]
        for transform in transforms:
            imgs.append(transform(image=img)['image'])

        imgs = torch.stack(imgs,dim=0)
        return imgs
        
class MixDataset(data.Dataset):
    """
    Using diff strategy to mix the dataset \n
    Now we support those:
    
    - basic rebalance, complete merge
    
    Need Support in the future:
    - diff downsample strategy such as prototypical
    
    """
    def __init__(self, dataset, new_dataset, strategy='balance', factor=1, 
                    transformer=None, augs=None, *args, **kwargs):
        
        self.dataset = dataset
        self.transformer = transformer
        self.new_dataset = new_dataset
        self.data, self.targets = copy.deepcopy(new_dataset.data), copy.deepcopy(new_dataset.targets)
        self.augs = augs
        
        # using specific strategy to mix the dataset
        # we may caculate the factor in the process
        self.select_stategy(strategy,factor)
        
        return None
        
    def __getitem__(self, index):
        label = self.targets[index]

        if isinstance(self.data[index], str):
            img = cv2.imread(self.data[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = self.data[index]

        if img.shape[0] == 3: img = img.transpose(1,2,0)
        
        # 
        if self.transformer is not None:
            try:
                img = self.transformer(img)
            except:
                img = self.transformer(image=img)['image']
        
        if isinstance(img, list):
            for pic in img:
                if not isinstance(pic, torch.Tensor):
                    pic = transforms.ToTensor()(pic)
                
        else:
            if not isinstance(img, torch.Tensor):
                img = transforms.ToTensor()(img)
            
        return img, label

    def __len__(self):
        return len(self.targets)
    
    def select_stategy(self, strategy, factor):
        # using thif function to call the related function
        if strategy == 'balance':
            self.mix_balance(factor)
        
        elif strategy == 'complete':
            self.data += self.dataset.data
            self.targets += self.dataset.targets    

        else:
            raise NotImplementedError("strategy not implemented")
        return None
    
    def mix_balance(self, factor):
        """        
        using factor to downsample the old dataset
        we need to modify this make it by class:
        
        1. sort by the index and select same num for each class
        2. shuffle and get the dats
        
        the sort function can be seprate by this files
        """
        sort_dataset(self.dataset)
       
        num_cls_old = len(np.unique(self.dataset.targets))
        steps = np.bincount(self.dataset.targets)
        
        # using the factor to balance the dataset
        # downsample the old dataset for now
        num_new = int(len(self.dataset) * factor) 
        num_new = round(num_new / num_cls_old)

        # make num of each class balance
        # random old class's data for each class

        # get the random mix of dataset. 
        index = 0
        for i in range(num_cls_old):
            idx = np.random.permutation(steps[i])
            selected_idx = idx[:num_new]
            for j in selected_idx:
                self.data.append(self.dataset.data[index+j])
                self.targets.append(self.dataset.targets[index+j])

            # using data augmentation 
            if len(selected_idx) < num_new:
                num_extra = num_new - len(selected_idx)

                for k in range(num_extra):
                    rd_idx = np.random.choice(selected_idx,1,replace=True)
                    new_img  = self.dataset.data[index+rd_idx[0]]
                    # new_img = self.transformer(new_img)
                    new_img = self.augs(image=new_img)['image']

                    self.data.append(new_img)
                    self.targets.append(self.dataset.targets[index+rd_idx[0]])
            
            index += steps[i]
        
        return None

if __name__ == "__main__":
    # test the whole process of this script
    
    # =========================================SEPRATE VERSION TEST============================
    # 1. get data by diff type 
    DATATYPE = 'cifar100'
    dataset = AllData(DATATYPE,istrain=0, save_path=r'/workspace/DownloadData')

    # # 2. visualize the origin output for debug
    print('dataset.class_num: {}'.format(len(dataset.targets)))
    
    # 3. generate the new dataset 
    # but in the future we may change the num of it, so we maybe do not setting a map
    # or write a function to producing it 
    sampler = DataSampler(dataset)
    new_dataset = sampler.get_new()
    update_dataset = sampler.get_remain()
    print('check new_class generator')

    # 4. make it unbalance
    imb_dataset = sampler.get_imb()
    print("Success with basic part of dataset")
    
    # 5. save the dataset and test read data for the npy 
    subfix = DATATYPE + '_' + 'new'
    sampler.save_dataset(new_dataset, r'/workspace/DownloadData', subfix)
    print('Success Save the new dataset')

    # 6. load the new dataset we generate after sampling
    res = AllData('LocalFiles', istrain=0, save_path=r'/workspace/DownloadData/cifar100_new')
    # ============================================INTEGRATE VERSION TEST=================