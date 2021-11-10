""" @Aiken 2021 
This file is coding to show every batch data we organized
1. sampling data and visualize it 
2. visualize those img after transforms
3. integrate those img into tensorboard for display and analysis

基本可视化可参考：https://blog.csdn.net/hester_hester/article/details/102516928
可视化保存代码参考：https://blog.csdn.net/disanda/article/details/90744243
"""
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from util.wraps import visual_1s
# 这个实际上是一个粗浅的可视化的模块，后续的话可能要添加一些统计的可视化值在这个代码文件中

@visual_1s(phaseName = "first_batches")
def dataVisualization(data,label,showtype=0):
    """可视化展示数据和对应的label

    Args:
        data : [可以有两种形式，有/无 batch size]
        label : [每张图片对应的标签]
        showtype : [展示方式，直接显示还是使用tensorboard之类的去现实]
    """
    # 将每一组照片集成到一个figure中，指定figure的显示大小
    res = []
    Len = len(data) if isinstance(data,list) else len(data.shape)
    # 对按照batchsize格式输入的图像的处理
    if Len >= 4 :
        b_s, n_channel = data.shape[0], data.shape[1]
        loop_counts, remain_nums = b_s // 5, b_s % 5
        value = data
        for i in range(loop_counts):
            fig = plt.figure(figsize=(11,3))
            for j in range(1,6):
                index = 5*i + j
                plt.subplot(1,5,j)
                # adapt the type of a image to what we need 
                value_grid = torchvision.utils.make_grid(value[index])
                value_grid = value_grid/2 + 0.5
                if isinstance(value_grid,torch.FloatTensor):
                    value_grid = value_grid.numpy()
                if n_channel == 1:
                    plt.imshow(value_grid,cmap='Greys')
                else:
                    try:
                        plt.imshow(np.transpose(value_grid,(1,2,0)))
                    except:
                        plt.imshow(value_grid)
                plt.title("label: {} ".format(label[index]))
            res.append(fig)
        
        for i in range(remain_nums):
            fig = plt.figure(figsize=(remain_nums*2,3))
            plt.subplot(1, remain_nums, i+1)
            index = loop_counts*5 + i
            value_grid = torchvision.utils.make_grid(value[index])
            value_grid = value_grid/2 + 0.5
            if isinstance(value_grid,torch.FloatTensor):
                value_grid = value_grid.numpy()
            if n_channel == 1:
                plt.imshow(value_grid,cmap='Greys')
            else:
                try:
                    plt.imshow(np.transpose(value_grid,(1,2,0)))
                except:
                    plt.imshow(value_grid)
            plt.title("label: {} ".format(label[index]))
        res.append(fig)

    else:
        figure = plt.figure()
        b_s, n_channel = 1, data.shape[0]
        value = data
        # if n_channel == 1:
        #     value = value.mean(dim=0)
        value =  value * 0.5 + 0.5
        npimg = value.numpy() if not isinstance(value, np.ndarray) else value
        
        if n_channel == 1:
            plt.imshow(npimg, cmap='Greys')
        else:
            plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.title("label: {} ".format(label))
        res.append(figure)

    # 建立窗口
    if showtype == 0:
        # 直接显示图像，这里没有设置多图像的显示所以目前只能一张一张的看
        plt.show()
    else:
        # 用于tensorboard的图像绘制部分的使用
        return res


def feature_visual(features, labels, writer, n_iter, *args, **kwargs):
    """
    可视化特征值
    Args:
        features: 特征值
        labels: 标签
    """
    # input: B-CHW, using torch.make_grid to a batch of images then add it in tensorboard
    imgs = torchvision.utils.make_grid(features, nrow=8, normalize=True, scale_each=True)
    writer.add_image('feature_map', imgs, n_iter)
    