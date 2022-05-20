""" @Aiken 2021
this file is about some data preprocesss;
some useful or necessary features for data processing;
# 参考了自己在EnAET的实验算法中的代码部分，
"""
import os
import glob
import csv
import cv2
import numpy as np
from tqdm import tqdm

from util.Visualize import dataVisualization
from sklearn.model_selection import train_test_split


# 将这一系列的不同处理写在同一个函数中的时候，就可以通过外部的接口来选择性调用其中的某一部分，这样是不是好一些？
DATASTRUCTURES = ['.csv','.json','.yaml','.xml']
DATATYPES = ['train','val','test']

# 将对于本地的数据文件可能会涉及到的操作都存放在这个地方，包括切分reshape等等一系列的操作
class LocaDataMethod:
    def __init__(self,imgpath=None,type=None):
        super(LocaDataMethod,self).__init__()
        if(imgpath != None):
            self.imgpath = imgpath
        if(type):
            self.type = type
        # 假设所有的图片都在同一个文件夹之中，通过glob获取所有图像文件的路径
        self.allimage = glob.glob(imgpath + '*')
        pass

    def resizeImage(self,outputsize = None, newAdress = None,*args,**kwargs):
        """对数据集中的图像进行resize的操作，未指定newAdress的时候就在原地址上进行操作

        Args:
            outputsize ([list]): [2-dim list, [500,500]]. Defaults to None.
            newAdress ([subfix]): [新地址]. Defaults to None.
        """
        if(isinstance(outputsize,list) == False): outputsize = [84,84]
        
        for i, img in enumerate(tqdm(self.allimage)):
            
            im = cv2.imread(img)
            im = cv2.resize(im,(outputsize[0],outputsize[1]),interpolation=cv2.INTER_CUBIC)
            # 设置新地址存储
            if(newAdress != None):
                p,f = os.path.split(img)
                img = newAdress + f

            cv2.imwrite(img,im)
            # 监控顺利输出的index罢了
            if (i%500 == 0):
                print(i)
        pass

    # 实际上这个函数是模拟mini-imagenet的方式坐的文件处理，也就是将文件名作为标签，
    # 但是实际上这种格式的csv已经能够当成dataloader来进行使用了把，直接对地址进行载入
    
    def MovSplitFiles(self,IMAGE_PATH = None,recorder_path = None, type = 0, aim_path = None):
        if(IMAGE_PATH == None): IMAGE_PATH = r"images/"
        
        # 将数据放入正确的directory中,
        for datatype in DATATYPES:
            if not (os.path.exists(recorder_path + datatype + DATASTRUCTURES[type])):
                continue
            os.mkdir(datatype)
            # 判断是否存在相应的val,由于不是所有的数据集都会且切分出验证数据集来的
            with open(recorder_path + datatype + DATASTRUCTURES[type] , 'r', encoding='utf-8') as f:
                read = csv.reader(f,delimiter = ',')
                error_label = ''
                for i, row in enumerate(read):
                    if(i == 0): continue
                    label = row[i]
                    imageN = row[0]
                    # 当有序排列我们的数据的时候，我们就能很简单的判断当前的文件夹是否是存在
                    if(label != error_label):
                        # 当存在合理标签的时候就将文件放到指定的地方去
                        cur_dir = aim_path + datatype + '/' + label + '/'
                        os.mkdir(cur_dir)
                        error_label = label
                    os.rename(IMAGE_PATH+imageN,cur_dir+imageN)
        pass

    def UniversalReader(self):
        # 占位函数，后续将movesplit函数的文件处理功能从这边继承
        return 
        
# 实际上后面两种写法都是一样的，所以我可以用*args将比较不常用的放在那
# 一般也就是使用dataloader

def samplingForDisplay(testloader,sampleType = 0, *args, **kwargs):
    """[using testloader(default) or output_data(arg,index =0) to display samples]

    Args:
        testloader 
        sampleType (int, optional): [1: output_data 2:testload 3:testload]. Defaults to 0.
    """
    datatype = "Dataset"
    # 首先可以看看kwargs中是否有我们需要的参数
    if 'datatype' in kwargs.keys():
        datatype = kwargs['datatype']
        print("now we'll display the data from {}".format(kwargs['datatype']))
    
    if sampleType == 1:
        try:
            output_data = args[0]
        except:
            print("donot have output_data")
        else:
        # type1 是对cifar100_data本身建立的迭代，没有对batch进行处理的
            for index,(data, label) in enumerate(output_data):
                # 只针对第一项输出数据类型。
                if index<1:
                    dataVisualization(data,label)
                    print('{} data_shape is like that: {}'.format(datatype, data.shape))
                    print('the first label is {}\n'.format(label))
                    break
            pass

    elif sampleType == 2 :
        # 这是最正统的写法，首先生成迭代器，然后按照batch输出下一项，只执行一次next
        dataiter = iter(testloader)
        data, label = next(dataiter)
        dataVisualization(data,label,showtype=1)
        
        # print('{} data_shape is like that: {}'.format(datatype,data.shape))
        # print('label shape is like that : {}'.format(len(label)))
        # print('the first label is {}'.format(label))

    elif sampleType == 3:
        for index,(data, label) in enumerate(testloader):
            # 只针对第一项输出数据类型。
            if index<1:
                # print('index: {}'.format(index))
                print('{}  data_shape is like that: {}'.format(datatype,data.shape))
                print('the first label is {}\n'.format(label))
        pass

    else:
        pass

def count_data(Ddict):
    """[return the classes_num and total data num]
    Args:
        Ddict ([type]): [dataset that stores as Dict]
    Returns:
        [list]: [0:num_class 1:total_num]
    """
    num_data, num_key = 0, 0
    for key in Ddict.keys():
        num_key += 1
        num_data += len(Ddict[key])
    return [num_key,num_data]

# function test part 
if __name__ == '__main__':
    with open(r'D:\LocWorkSpace\tempdata\mini-imagenet\train.csv', 'r') as f:
        read = csv.reader(f,delimiter=',')
        for row in read:
            pass
    
    # 主要到这里如果我们使用装饰器的话，带返回值和不带返回值实际上要调用的是两个函数
    # value = TestWrapper(5)
    # print(value)

    # str_time = time.time()
    # output_data, testloader = download_data(r'D:\local_CODE\tempdata', is_training= True, type=0, datatype='cifar100', shuffle = True)
    # # download_data(r'D:\local_CODE\tempdata',is_training=True,type=0,datatype='cifar100')
    
    # print(output_data.data[1].shape)
    # # print(testloader.shape)
    # dataiter = iter(testloader)
    # data, label = dataiter.next()
    
    # # ------------------------------------npy test------------------------------------
    # # -------------------------success&next step we need to use those data
    # creat_npy_dataset(output_data,r'D:\local_CODE\tempdata','cifar100_test',30,False)
    # image_grid = torchvision.utils.make_grid(data)
    # showtype = 1  #控制是否使用tensorboard
    # if showtype != 0:
    #     writer = SummaryWriter('runs/cifar100_experiment_1')
    #     fig = dataVisualization(image_grid,label,showtype=showtype)
    #     # writer.add_image('test_image', image_v2)
    #     writer.add_figure('testfig',fig)
    #     writer.close()    
    # else:
    #     # dataVisualization(image_grid,label,showtype=showtype)
    #     pass
    

    
    # end_time = time.time()·
    # print('run time : ',end_time-str_time)
