""" @Aiken 2021 Metrics Part
topn-ACC|F1|Precise|RECALL
"""
from sklearn import metrics as skm
from util.utils import * 
from util.wraps import *

# 全局变量解包

# 在每个Training Process 创建一个Recorder，指定需要的Evaluatator，以及传入相应的数据
# 每个batch新载入的数据，按照我们需要的Evaluator：

# 进行计算和更新相应的指标（进行累加）
# 装饰器实际上应该是放在这个evalutor.update/output 这里，在指定更新的时候将数据传入tensorboard

# 有一个问题是这里的count目前定义为通用的形式，所以如果要区分过程的话，我们就使用不同的recoder进行
class Recorder():
    def __init__(self, rec_dict = None, **kwargs):
        super(Recorder, self).__init__()
        self.recorder = {}
        self.exceptKey = ['confi']

        self.count = 0
        
        self.recorder['count'] = self.count
        self.recorder['epoch'] = 0
        self.recorder['itera'] = 0
        # we must know how many iterations we have run, which can calculate the real avg 
        # init all the parameters by rec_dict and update lately
        if rec_dict is not None:
            
            self.count += 1 
            self.recorder['count'] = self.count

            for k,v in rec_dict.items():
                self.recorder[k] = v 
            for k,v in kwargs.items():
                self.recorder[k] = v 
        
        
    @record_nbatch(verbose=False)
    def update(self, rec_dict, **kwargs):
        # 通用的输出模块
        res = {}

        # 对于传入的参数直接进行输出然后对于字典的值进行累加
        self.count += 1
        self.recorder['count'] = self.count
        # we need to except some data we donot want to grad
        # update dict by dict:key-value, which make the info more reliable 
        res['count'] = self.count
        
        for k,v in rec_dict.items():
            res[k] = v
            if k in self.exceptKey: continue
            if self.recorder.get(k):
                self.recorder[k] += v
            else:
                self.recorder[k] = v
            
        # combination rec_dict with kwargs  
        for k,v in kwargs.items():
            res[k] = v
            if k in self.exceptKey: continue
            if self.recorder.get(k):
                self.recorder[k] += v
            else:
                self.recorder[k] = v 
             
        res['epoch'] = self.recorder['epoch']
        res['itera'] = self.recorder['itera']

        return res 
    
    @record_nbatch(verbose=True)
    def final(self, doavg = True):
        if not doavg: return self.recorder
        
        temp = self.recorder['count']
        temp_e = self.recorder['epoch']
        
        for k,v in self.recorder.items():
            self.recorder[k] = v / self.count

        self.recorder['count'] = temp
        self.recorder['epoch'] = temp_e
        return self.recorder
    
    def reset(self):
        temp = self.recorder['epoch']
        
        self.recorder.clear()
        
        self.recorder['epoch'] = temp + 1
        self.recorder['itera'] = self.count

        self.count = 0
    
    def addExcept(self,keys):
        if keys in self.exceptKey: return

        if isinstance(keys,list):
            self.exceptKey += keys
        else:
            self.exceptKey.append(keys)
        
class Metric_cal():
    def __init__(self):
        super(Metric_cal,self).__init__()
        # output parameters
        self.out_dict = {}
        self.types = ['acc1','acc5','prec','recall','f1']
    
    def __call__(self,label,pred,needtype):
        # parameters inplement 
        self.labels = label
        self.pred = pred
        acclist = []
        # using type to control metrics
        # 这部分还可以进行优化，怎么对acc的方式进行整合
        for type in needtype:
            if(type not in self.types): raise NotImplementedError('not supposed this metric: {}, please add it in Metric _cal'.format(type))
            if(type == 'acc1'):
                acclist.append(1)
                # res = topn_acc(self, self.labels, self.pred, (1,))
            elif(type == 'acc5'):
                acclist.append(5)
                # res = topn_acc(self, self.labels, self.pred, (5,))
            else:
                raise NotImplementedError
                res = skl_metrics(type)
                for k,v in res.items():
                    self.out_dict[k] = v
        
        if len(acclist) != 0:
            res = topn_acc(self.labels, self.pred, acclist)
            for k,v in res.items():
                self.out_dict[k] = v
        
        # return dict with metrics key
        return self.out_dict
    
    def calculate_each_acc():
        return None
    
    @text_in_tensorboard(Title='The PR of confi filter')
    def confi_metric(self, new_labels, new_total, num_news=79, verbose=True):
        """
        desc: calculate the PR for the confidece filter, mapping label to new and old 
        param: 
            new_labels: the collections of datas which is be filtered by confidence
            new_total: the nums of all the new datas
            num_news: the num_news to set new or old
        return:
            the PR of the confidence filter, this will be add in the tensorboard
        """
        # set the format of output 
        fmt = "Precisions: {:<.3f}  \t  \n Recall: {:<.3f}  \t  \n"
        
        # change the data status on cpu
        
        c_new_labels = torch.cat(new_labels).cpu().numpy()

        # calculate the PR
        precision = len(np.where(c_new_labels>num_news)[0])/len(c_new_labels)
        recall = len(np.where(c_new_labels>num_news)[0])/new_total

        # output the PR
        res = fmt.format(precision,recall)
        if verbose: print(res)
        
        return res
        
    
    @text_in_tensorboard(Title="new_cls_confi")
    def calculate_each_confi(self, num_cls, pr_res_value, pr_labels, ):
        """
        using decorator to add the result 
        """
        meanv = 1./ num_cls
        pr_res_value.squeeze_(dim=1)

        belong_beyond_cls = pr_labels >= num_cls
        confi_max = [pr_res_value[i] for i in range(len(belong_beyond_cls)) if belong_beyond_cls[i]]
        confi = [(confi_max[i]-meanv)/(1-meanv) for i in range(len(confi_max))]

        from itertools import groupby
        confi.sort()
        confi_group_by = " "
        for k, g in groupby(confi, key=lambda x: x//0.1):
            confi_group_by += "{:<.3f}-{:<.3f} : {}  \t  \n".format(k*0.1,(k+1)*0.1,len(list(g)))
        if True:
            print(confi_group_by)

        return confi_group_by

def topn_acc(labels, pred, topn =(1,5), use_skl = False, **kwargs):  
    # ! 这里注意，输入的Label标准形式而不是onehot的形式，如果需要加入支持的话，我们可能需要进行额外的转化
    # 但是通常情况下不需要这一点
    batch_size = pred.size(0)
    key_name = 'acc'
    # According to the max requirement to get index and value 
    max_req = max(topn)
    _,index = pred.topk(max_req,1,True,True)
    out_dict = {}
    count, acc = [], [] 
    if not use_skl:
        # we need to expand the dimension to compare the resule 
        corr = index.eq(labels.view(-1,1))
        
        # collection of topn: this is for the following processing 
        for index,i in enumerate(topn):
            corr_k = corr[:,:i].view(-1).float().sum()
            count.append(corr_k)
            acc.append(corr_k/batch_size)
            # 加入输出列表
            tempkey = key_name + str(i)
            out_dict[tempkey] = acc[index]
    # using sklearn to return the result of acc and acc count 
    else:
        # 在这里输出的已经是按照batch取完平均后的结果了
        for i in topn:
            res = skm.top_k_accuracy_score(labels,pred,k=i)
            acc.append(res)
            count.append(res*batch_size)
            # 加入输出列表
            tempkey = key_name + str(i)
            out_dict[tempkey] = acc
    
    return out_dict

# we need to judge this module in the future. or rewrite this part(donot use skm)
def skl_metrics(labels,pred,*args):

    # intergrate some metrics from sklearn module
    out_dict = {}
    if 'prec' in args:
        out_dict['prec'] = skm.average_precision_score(labels,pred,average='samples')
    if 'recall' in args:
        out_dict['recall'] = skm.recall_score(labels,pred,average = 'samples')
    if 'f1' in args:
        out_dict['f1'] = skm.f1_score(labels,pred,average = 'samples')
    return out_dict

# test this module 
if __name__ == '__main__':
    # SKLEARN的metric 使用起来问题太多了，还不如自己写的topn
    # define the test parameters 
    # not one hot type and ont hot type 

    # Generate test data 
    import torch 
    batch,num_class = 4, 6

    label = torch.randint(0,num_class,(batch,1))
    print("==> label is : \n", label, '\n')

    # one_label = make_onehot_array(num_class,label)
    # print("==> one-label is : \n", one_label, '\n')

    pred = torch.rand(batch,num_class)
    pred = torch.softmax(pred,dim =1)
    print("==> pred is : \n ", pred, '\n')

    _,index = pred.topk(5,1,True,True)
    print("==> pred_index is : \n ", index, '\n')

    # define metric class 
    Judge = Metric_cal()

    # not one hot 
    res = Judge(label, index, ('acc1','acc5'))
    print("==> not one res is : \n ", res, '\n')
    # skm.top_k_accuracy_score(label,index,k=1)

    
    