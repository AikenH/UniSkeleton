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


# ---------------------------------npy存储示例（非常不标准）------------------------
# 以下的示例存在比较多的问题，比如说这里的label和data完全可以用字典的形式存放在同一个npy中
# 我们可以将一个数据和其很多其他的info放置在一起，甚至是某个阶段中提取出来的特征，
# 这里就是一个基本方法的使用，后续可以针对自己的需求去写一个读写npy的过程

# Specific Infomation: this method is used to running FSL Process,and creat only run once
# we store the label and data into different place to resampling and split those data 
# but in fact, if we split data and transformer those data in advance, we can test faster
# (Using more hard drive space for faster runing )
# @timeCounterN
def creat_npy_dataset(output_data, prefix, suffix, max_num, is_training=True):
    '''在初始下载数据的时候将数据结构转化为npy，方便后续的读取
    prefix:基本地址  suffix：类别名称 max_num：最多转化多少数据'''
    # 当第一次执行数据集下载的时候将输出以npy的形式保存，提升后续数据os的速度

    basic_path = os.path.join(prefix, suffix)
    if not os.path.exists(basic_path):  os.mkdir(basic_path)
    sub_path = os.path.join(basic_path,'trainset') if is_training else os.path.join(basic_path,'testset')
    if not os.path.exists(sub_path): os.mkdir(sub_path)

    num_file = len(os.listdir(sub_path))/2
    if num_file<max_num:
        # 计算还差多少个文件，别多生成了,(默认是目标和数据匹配的)
        # 后续可以补充的就是判断其中label和train文件的数目是否一致，如果不一致返回Error吧
        epoch = min(max_num,len(output_data.targets))
        basenum = int(num_file) 
        # 这里的label和相应的data为什么不一起存储呢？
        for i,(data,label) in enumerate(tqdm(output_data)):
            if 'trainset' in sub_path: 
                temp_data_path = os.path.join(sub_path,'trainset'+str(i+basenum)+'.npy')
            else:
                temp_data_path = os.path.join(sub_path,'testset'+str(i+basenum)+'.npy')

            temp_label_path = os.path.join(sub_path,'labelset'+str(i+basenum)+'.npy')
            # output.data这种操作好像不行？

            # dataVisualization(data,i,showtype=0)
            np.save(temp_data_path,data)
            np.save(temp_label_path,label)
            
            # break 
            if i+basenum == epoch-1:
                print('it time to stop : ', i+1, 'iteration we make')
                break 
    else: print('thera are enough data, no more needed!!!')

    num_file = len(os.listdir(sub_path))/2
    print('after all operation, there are {} files '.format(num_file))

# 对应的文件处理模块，实际上读取的时候就使用np.load()就可以了
def get_four_npydatapath(dataset_dir,proportion,v_rate,directly=False):
    '''基于npy文档读取数据的方式,分别输出指定的train（label，unlabel）， test（test，varify）
    对应的npy 所在的地址的list 
    proportion:分割标注和未标注的比例, 验证集和测试机的比例'''
    temp_pathlist = [os.path.join(dataset_dir,'trainset'), os.path.join(dataset_dir,'testset')]
    output_list = []
    for i,temp_path in enumerate(temp_pathlist):
        listfiles = os.listdir(temp_path)
        if  i == 0:
            all_data = [os.path.join(temp_path, x) for x in listfiles if 'trainset'  in x]
            # plan b 还可以进一步改进这一部分,只留下数字,但是八成是没必要了吧,如果只留下数字的话,直接用迭代器迭代有什么差别呢.
            # all_data = [x for x in listfiles if 'trainset' in x]
        else:
            all_data = [os.path.join(temp_path, x) for x in listfiles if 'testset' in x]
            # plan b 
            # all_data = [x for x in listfiles if 'testset' in x] 
        
        all_label = [os.path.join(temp_path,x) for x in listfiles if 'labelset' in x]
        # planb
        # all_label = [x for x in listfiles if 'labelset' in x]
        # 排序数据的目的是为了让图像和标签能对齐
        all_data.sort()
        all_label.sort()
        # part1:拆分数据集
        # add_output 
        
        if i == 0:
            x_unlabeled, x_labeled, y_unlabeled, y_labeled = train_test_split(all_data,\
                all_label,test_size=proportion,random_state=888)
        else:
            x_varify, x_test, y_varify, y_test  = train_test_split(all_data,\
                all_label,test_size=v_rate,random_state=888)

    if directly:
        return x_labeled,y_labeled,x_unlabeled,y_unlabeled,x_test,y_test,x_varify,y_varify,all_data,all_label
    else:
        return x_labeled,y_labeled,x_unlabeled,y_unlabeled,x_test,y_test,x_varify,y_varify