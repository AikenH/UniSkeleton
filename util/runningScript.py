""" @Aiken 2021 April
This File is Write to manage the Running Process:
including the Traing Testing Varifying.
"""
# Basic Module
import os
import copy
import torchvision
import tqdm
import logging
# Torch module
import torch
from torch import nn,optim
from torch.utils import data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

# Local module
from util.utils import *
from util.wraps import *
from util.metric import Recorder,Metric_cal
from data.dataUtils import *
from data.datasetGe import AllData, DataSampler
from model import Assembler
from loss import L_select as loss_setter
from layers.ezConfidence import leastConfi
# acceleate module
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

# FIXME: we need to enable the test loader in the test part, in this version, we using val as our test dataset
# ATTENTION: if u want to write new training process, we inherit this class
# then write your own training process to override the train and test function
class base_runner():
    def __init__(self,config):
        super(base_runner, self).__init__()
        # loading based parameters & parser some important parameters
        self.config = config
        self.model_t = config['networks']
        
        self.val_n = config['val_n']
        self.resume = config['resume']
        self.num_cls = config['data']['dataset']['num_cls']

        # loading training opt dict for training process
        self.training_opt = self.config['training_opt']
        self.dataset_opt = self.config['data']['dataset']
        self.freq = self.training_opt['freq_out']
        self.dataopt = self.training_opt['dataopt']
        self.drop = self.training_opt['train']['drop_rate'] 
        
        # init recorder for train and val 
        self.train_rec = Recorder()
        self.val_rec = Recorder()
        self.test_rec = Recorder()
        
        # set metric here or in the yaml files 
        # init metrics for train and val
        self.Metrics_tool = Metric_cal()
        self.metrics_type = config['metrics_type']
        # self.Metrics_type = ['acc1', 'acc5']

        # self.ckpt_pth = config['ckpt_pth']
        # initialize the logger without parameters ,prefix
        # (we can change this in the future)
        # (or move this part to evaluation or training, nop)
        logger_dict = self.init_logs(**config['logger'])
        self.logger = logger_dict['logger']
        self.writer = logger_dict['writer']
        self.t_pth = logger_dict['tPth']
        
        # call the func to import apex module
        # self.apex_flag = try_import_apex(self.config['apex'])
        self.apex_flag = config['apex']
    
    #  =============================================== MAIN FUNCTION =================================
    @log_timethis(desc = "training ")
    def train(self):
        
        # 0.load data from data.py(or self)
        # refer to others' solution and consider how to import those data
        # specifically, rethink whether the way we write dataloader is resonable
        dataset = AllData(**self.dataset_opt)
        data_tag = self.config['data']['preprocess']        

        if data_tag:
            TRAIN_DATASET = DataSampler(dataset, **self.dataset_opt)
            TRAIN_DATASET.process()
            dataset = TRAIN_DATASET.imb_dataset if data_tag != 'new' else TRAIN_DATASET.remain_dataset
            #ATTENTION: if we want to use this dataset, we need to specific deal with it
            new_dataset = TRAIN_DATASET.new_dataset if data_tag != 'imb' else None
            if data_tag != 'imb': self.new_dataloader(new_dataset, **self.dataopt)
            
        self.dataloader = data.DataLoader(dataset, **self.dataopt)
        # change the satus to test 
        self.dataset_opt['istrain'] = 1
        test_dataset = AllData(**self.dataset_opt)
        self.test_dataloader = data.DataLoader(test_dataset, **self.dataopt)

        # change the status to val
        self.dataset_opt['istrain'] = 2
        val_dataset = AllData(**self.dataset_opt)
        self.val_dataloader = data.DataLoader(val_dataset, **self.dataopt)

        # visualization images for the first batch 
        samplingForDisplay(self.dataloader,sampleType=2)
        
        # 0 init model,optimizer,loss by function and config
        # 这一步后续是需要被完善的
        self.main_model = self.init_model()
        
        # 1. transfer optimizer and loss to CUDA
        optim_ = self.init_optim(self.training_opt['optimizer'])

        opt_dict = self.training_opt['optim_opt']
        sche_dict = self.training_opt['sche_opt'] 
        
        self.optimizer = optim_(self.main_model.parameters(),**opt_dict)
        schedule_t = self.training_opt['schedule']
        schedule_f = self.init_schedule(schedule_t)

        if schedule_t == 'warmup':
            # define a function here or in advance, FIXME ↓
            warmup_With_Step = lambda epoch: ...
            self.schedule = schedule_f(self.optimizer,warmup_With_Step,**sche_dict)
        else:
            self.schedule = schedule_f(self.optimizer,**sche_dict)

        # LOSS_V0 (only one criter) calculate the loss and update model parameters
        self.criterion = self.init_loss(self.training_opt['loss'])
        if torch.cuda.is_available(): self.criterion = self.criterion.cuda()

        # using amp to speed up the training
        if self.apex_flag:
            self.main_model, self.optimizer = amp.initialize(self.main_model, self.optimizer, opt_level ="O1")

        s_ep = 1
        # 2. resume model(ckpt) or get parameters from pretrain model 
        if self.resume:
            resumeDic = self.load_model_v0(self.main_model, self.optimizer,self.config['resume_type'])
            
            # 恢复writer，重新设置writer，恢复epochs，恢复recorder中的epoch、itera
            s_ep = resumeDic["i"] if resumeDic.get("i") else s_ep
            self.train_rec.recorder['epoch'] = s_ep 
            self.train_rec.recorder['itera'] = resumeDic["train_iter"]
            self.val_rec.recorder['epoch'] = int(s_ep/self.val_n)
            self.val_rec.recorder['itera'] = resumeDic['val_iter']

            if resumeDic.get('log_path'):
                self.writer = SummaryWriter(resumeDic['log_path'])
                self.train_rec.update.set_writer(self.writer)
                self.train_rec.final.set_writer(self.writer)
                dataVisualization.set_writer(self.writer)
        
        # NOTE: 加载预训练模型的话，就是重新初始化logger和writer
        # Placeholder: load pretrain model.

        '''======================================TBF================================'''
        # 3. start training process

        self.train_epoch(self.training_opt['train']['num_epochs']+1, s_epoch=s_ep, val_n=self.val_n)

        # evalutaion-output-save model in training + evalutaion
        # 4. test on the test dataset or test dataset and output the final result 
        self.test()
        
        '''=======================================TBF================================'''
        # 5.close the logger and writter 
        if self.writer != None:
            self.writer.close()
    
    # 在这一块定义recorder和调用相应的metric，同时在这里返回每个epoch的输出，然后用Recorder2 输出这一部分
    def train_epoch(self,epoches,s_epoch = 1,val_n = 10):
        EARLY_COUNT = 0
        EARLY_STOP = self.training_opt['train']['early_stop']
        loss_h,acc1_h,acc5_h = None,None,None
         
        for i in range(s_epoch,epoches):
            self.train_rec.update.set_phase('training_')
            
            self.main_model.train()
            # run n batchs for each epoch 
            # the infomathion will be shown by decorator in train_nbatch()
            self.train_nbatch(i, iterations = self.training_opt['train']['iterations'] \
                            if self.training_opt['train'].get('iterations') else None)
            
            # 更新学习率的时候添加学习率到可视化的部分

            self.writer.add_scalars('Weights/lr',{'lr': \
                                    self.optimizer.param_groups[0]['lr']},i)
            self.schedule.step()
            
            self.train_rec.final.set_phase('train_epoch_')
            self.train_rec.final()
            self.train_rec.reset()
            
            # visualize those parameter as historgram
            # we can add other model here
            if i % 10 == 0:
                for name,param in self.main_model.named_parameters():
                    self.writer.add_histogram('main_model'+name,param.clone().cpu().data.numpy(),i)
                pass

            # SAVE: every k batch, judge whether the current model is better, and then update the result(save)
            # EARLY-STOP(ES): at the same time we analyze whether we need to break out of this loop to avoid useless traing
            if val_n is not None and i%val_n == 0:
                
                self.val_rec.update.set_phase('valing_')
                # update the best result for SAVE and EARLY_COUNT
                result_val = self.varify_v0('val')
                
                loss_h,symbol_l = new_min_flag(loss_h,result_val['loss'])
                acc1_h,symbol_a = new_max_flag(acc1_h,result_val['acc1'])
                acc5_h,_ = new_max_flag(acc5_h,result_val['acc5'])
                
                #  SAVE: using evalute like acc to save best model
                if symbol_a:
                    subfix = self.t_pth.split('/')[-2]
                    self.save_model_v0(self.main_model, self.optimizer, i=i, subfix=subfix,log_path=self.t_pth)

                # After Saving models
                # # ES: using loss indicator to stop training 
                # if(symbol_l is False): EARLY_COUNT = 0
                # EARLY_COUNT += 1 if symbol_l else 0

                # ES: using acc indicator to stop training 
                if symbol_a == False: EARLY_COUNT = 0
                EARLY_COUNT += 1 if symbol_a else 0

                if(EARLY_COUNT == EARLY_STOP):
                    self.logger.info('> !EARLY-STOP!{} END TRAINING <'.format(EARLY_COUNT) )
                    break
        

        self.logger.info('> END of TRAINING <')
 
    def train_nbatch(self, cur_epoch, iterations=30):
        # each epoch may run n batch，and we will upload info to tensorboard every batch
        self.logger.info('> EPOCH {} <'.format(cur_epoch) )
        Loss = 0
        # I_Loss = 0
        
        # ATTENTION：training with the iterations setting, but we donot use this in normal situation
        """ for i in tqdm(range(iterations),desc='EPOCH: {} '.format(cur_epoch)):
        
            get data and label
            try:
                img,label = next(self.dataiterater)

            except StopIteration:
                self.logger.info('!!!run out of data ,reloader iterator')
                self.dataiterater = iter(self.dataloader)
                img,label = next(self.dataiterater) """

        for i,data in enumerate(tqdm(self.dataloader, desc='EPOCH: {} '.format(cur_epoch))):
            # Load data which should be with the status: __getitem__ in dataset class
            img,label = data[0], data[1]

            # if we do not duplicate img with diff augmentation, we use this.
            
            # Extra Data Augmentation and label transformation.
            
            # CUDA
            if torch.cuda.is_available():
                # target = target.cuda()
                label = label.cuda()
                img = img.cuda()
            
            # # FIX:visualize model in tensorboard (bug in pytorch)
            # if cur_epoch == 1 and i == 1:
            #     self.writer.add_graph(self.main_model,(img,))
            
            # PLACEHOLDER: visulize some data if we want to varify data transformation or sth. else
            
            # using model to make prediction and conpare with labels
            if self.model_t['cls_model'] == 'Defaults' or self.model_t['feature_vis'] == False:
                pred = self.main_model(img)
            else:
                _, pred = self.main_model(img)

            # NOTE: if we want add visulization of heat map, we can write it here
            # maybe put this part in n_batch will be better?
            # read those data we want to visulize in here and call the function we need.

            Loss = self.criterion(pred,label)
            # using not zero loss to import the generalization of model
            pre_set_loss = self.training_opt['notzeroloss']
            if pre_set_loss is not None: 
                Loss = (Loss - pre_set_loss).abs() + pre_set_loss

            self.optimizer.zero_grad()
            if self.apex_flag:
                with amp.scale_loss(Loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                Loss.backward()
            self.optimizer.step()

            # try:
            train_res_dict = self.Metrics_tool(label,pred,self.metrics_type)
            
            # except:
                # print('pred: {} \n label {} \n loss {}'.format(label, pred, Loss))
                # print('the shape of those parameters is {} \n {} \n {} \n'.format(label.shape,pred.shape,Loss.shape))
                # raise TypeError('fix this problem')

            # Then the decorator will pass it to tensorboard and .log
            if(i % self.freq == 0):
                confidence = leastConfi(pred)
                train_res_dict['confi'] = confidence
                self.train_rec.addExcept('confi') # 该key只进行限时，不进行累计，因为累计没有意义
                self.train_rec.update(train_res_dict,loss = Loss.item())
        
        torch.cuda.empty_cache()
        
        return self.train_rec

    def test(self):
        self.test_rec.update.set_phase('test_')
        testres = self.varify_v0('test','vis')
        return testres
        
    def varify_v0(self,*args):
        # choose dataset from test and val set.
        idxs = 0
        if 'val' in args:
            RecorderV = self.val_rec
            dataloader_this = self.val_dataloader
        else:
            # if there is not val , the datasetge will using test as val
            RecorderV = self.test_rec
            dataloader_this =self.test_dataloader
            # using this part to generate random batch for projections 
            vis_idxs = torch.randperm(len(dataloader_this))[:5]
            vis_datas, vis_labels, vis_imgs = [], [], []
            pr_labels, pr_probs = [], []
        
        # change model status 
        self.main_model.eval()

        # using model to get result 
        self. logger.info('> Start Valing  <')
        
        with torch.no_grad():
            for img, label in tqdm(dataloader_this):
                if torch.cuda.is_available(): 
                    idxs += 1
                    label = label.cuda()
                    img = img.cuda()
                
                if self.model_t['cls_model'] == 'Defaults' or self.model_t['feature_vis'] == False:
                    pred = self.main_model(img)
                else:
                    feature,pred = self.main_model(img)
                    # 如果是attention这种heatmap的话可以直接再在这里可视化，不然的话，要我们自己去做聚类之类的处理
                    # features_img = torchvision.make_grid(feature,normalize=True, scale_each=True)
                    # self.writer.add_image('feature_map',features_img,self.train_rec['epoch'])

                    # show the embeeding of the features with tensorboard
                    if 'test' in args and idxs in vis_idxs:
                        vis_datas.append(feature)
                        vis_labels.append(label)
                        vis_imgs.append(img)
                    pass
                
                if 'test' in args:
                    pr_labels.append(label)
                    pr_probs.append(pred)
                
                # calculate those metrics for test and valling 
                Loss = self.criterion(pred,label)
                val_rec_dict = self.Metrics_tool(label,pred,self.metrics_type)
                confidence = leastConfi(pred)
                val_rec_dict['confi'] = confidence
                RecorderV.addExcept('confi') # 该key只进行限时，不进行累计，因为累计没有意义
                
                # update the metrics, and when we call the update and final the decortor will setting up tensorboard
                RecorderV.update(val_rec_dict, loss = Loss)

        if 'val' in args:
            RecorderV.final.set_phase('valing_epoch_')
        else:
            RecorderV.final.set_phase('test_epoch_')
        
        if 'test' in args and 'vis' in args and self.model_t['cls_model'] != 'Defaults':
            # add projectors here
            # concat those batches of data and label
            vis_datas = torch.cat(vis_datas,dim=0)
            vis_labels = torch.cat(vis_labels,dim=0)
            vis_imgs = torch.cat(vis_imgs,dim=0)

            # resize the tensor to a vector
            vis_datas = vis_datas.view(vis_datas.size(0),-1)
            # self.writer.add_embedding(vis_datas, vis_labels, vis_imgs)

            # add pr curve here.
            pr_labels = torch.cat(pr_labels,dim=0)
            pr_probs = torch.cat(pr_probs,dim=0)
            pr_probs = torch.stack([F.softmax(value,0) for value in pr_probs])
            for global_step in range(self.num_cls):
                tensor_labels = pr_labels == global_step
                tensor_probs = pr_probs[:,global_step]
                # FIXME :when we use the same name, the global step dones work for some reason
                self.writer.add_pr_curve('pr_class_{}'.format(global_step),tensor_labels,
                                                    tensor_probs,global_step=global_step)
            
        res = copy.deepcopy(RecorderV.final())
        
        RecorderV.reset()
        return res

    #  =============================================== BUILD A MODEL =================================
    # 如果需要自定义class的比如model或者loss，就按照再__init__中写setup的方法去做
    # 否则的话直接像select_optim 那样的写法去写就行了
    
    def init_model(self, *args, **kwargs):
        # 初始化模型，初始化损失函数初始化优化器
        model = Assembler(**self.model_t)
        # 读取并输出模型参数数目
        self.logger.info("the MODEL-{} have : {} PARAMETERS".format(
                        self.model_t['feat_model'],count_params(model)) )
        
        # 转换Cuda和Distribution
        if self.config['cuda_devices'] == '-1': return model
        if torch.cuda.is_available():
            if len(self.config['cuda_devices'].split(',')) > 1:
                if self.apex_flag:
                    model = DDP(model)
                else:
                    model = nn.DataParallel(model)
            model = model.cuda()

        img = torch.rand(4,3,224,224).cuda()
        self.writer.add_graph(model,img)
        return model

    def init_loss(self, loss_t):
        # 选择损失函数进行初始化，在Train中进行损失函数的设置和对全局的损失变量进行赋值
        loss = loss_setter(loss_t)
        self.logger.info("using {} LOSS<".format(loss_t))

        # 将损失转移到CUDA上
        if self.config['cuda_devices'] == '-1': return loss
        if torch.cuda.is_available():
            loss = loss.cuda()
        
        return loss
    
    def init_schedule(self, schedule_t:str):
        self.logger.info('using {} lr_schedule<'.format(schedule_t) )
        # 设置学习函数下降
        if schedule_t == 'none':
            return None
        elif schedule_t == 'exp':
            return optim.lr_scheduler.ExponentialLR
        elif schedule_t == 'cos':
            return optim.lr_scheduler.CosineAnnealingLR
        elif schedule_t == 'step':
            return optim.lr_scheduler.StepLR
        elif schedule_t == 'warmup':
            return optim.lr_scheduler.LambdaLR
        elif schedule_t == 'multistep':
            return optim.lr_scheduler.MultiStepLR
        else:
            raise NotImplementedError('NOT IMPLEMENTED, add in {}'.format(self.init_schedule.__name__))

    def init_optim(self, optimizer_t):
        """返回一个优化器类别，基于该优化器类别进行再初始化就能得到相应的optimizer，
        还需要进行完善，后续可能需要添加更多优化器，甚至进行优化器的自定义

        Args:
            optimizer_t ([type]): [全称描述的优化器名称]
        Returns:
            class: [需要重新初始化的优化器类别]
        """
        # NOTE：python中这种类似函数指针的返回方式是通用的还是需要重新进行集成
        # 最好将模型和参数在这个switch module 中也编写好，因为对于不同的优化器可能需要的参数是不一样的
        self.logger.info('using {} optimizer'.format(optimizer_t) )
        if optimizer_t == 'SGD':
            optim_ = optim.SGD
        elif optimizer_t == 'Adam':
            optim_ = optim.Adam
        elif optimizer_t == 'RMSProp':
            optim_ = optim.RMSprop
        elif optimizer_t == 'Adadelta':
            optim_ = optim.Adadelta
        else:
            raise NotImplementedError('please add this optim in the function: {}'.format(self.init_optim.__name__))

        return optim_

    #  =============================================== LOAD and SAVE =================================
    def load_model_v0(self,model, optimizer,type='ckpt',subfix=None):
        """ load ckpt or pretrain model part, single model 
            NOTE：对于组合型的model，这里暂定切分ckpt的存储然后通过循环调用来执行load model

        Args:
            desc (str, optional): Defaults to 'ckpt'.
        """
        # assert optimizer is not None, 'optimizer must be defined first~!'
        # assert model is not None, 'model must be defined first~!'

        # ================================= PATH COMPLETE ==============================
        ckptpth = self.config.get('ckpt_pth')
        assert os.path.exists(ckptpth), 'you need a ckptpath, if u want resume,or load pretrain model'

        # ================================= load universe parameters ================================= 
        checkpoint = torch.load(ckptpth)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if self.apex_flag:
            amp.load_state_dict(checkpoint['amp'])

        # ================================= load and specific parameters ================================= 
        if checkpoint.get('i'):
            self.START_EPOCH = checkpoint['i']
        
        if checkpoint.get('log_path'):
            self.log_path = checkpoint['log_path']
        
        return checkpoint
    
    @log_timethis('savemodel')
    def save_model_v0(self, model, optimizer,i=None, log_path=None,subfix = None):
        # ================================= setting ckpt to save =================================
        ckpt = {}
        ckpt['model'] = model.state_dict()
        ckpt['optimizer'] = optimizer.state_dict()
        if self.apex_flag:
            ckpt['amp'] = amp.state_dict()

        if i is not None:
            ckpt['i'] = i 
        if log_path is not None:
            ckpt['log_path'] = log_path
        
        ckpt['train_iter'] = self.train_rec.recorder['itera']
        ckpt['val_iter'] = self.val_rec.recorder['iter']
        # ================================= setting path ========================================
        now = time.strftime('%Y-%m-%d_%I:%M:%S %p')
        # Update the way we save model and edit code in this part
        if not subfix: 
            subfix = self.config['data']['dataset']['datatype'] + " " + now
        
        ckpt_pth = self.config.get('save_pth')
        ckpt_pth = os.path.join(ckpt_pth, 
                                self.model_t['cls_model'] + "_" +\
                                self.model_t['feat_model'])
        if not os.path.exists(ckpt_pth): 
            os.mkdir(ckpt_pth)
        ckpt_pth = os.path.join(ckpt_pth,subfix +'.pt')
        
        # ================================= saving it ============================================
        torch.save(ckpt, ckpt_pth)
        self.logger.info('> SAVE MODEL SUCCESSFUL {} <'.format(i) )
        return ckpt

    #  =============================================== utils and initialize  =================================
    def init_logs(self, logname = 'TrainLoger', log_dir='./log', console_level = 2, prefix = None):
        """ create logger to save all the info we want to show

        Args:
            logname (str, optional): Defaults to 'TrainLoger'.
            console_level (int, optional): Defaults to 3.
            prefix(str, optional): Set up experiment name here to store, if None, we make it TimeStamp
                                Defaults to None. 
            DEFAULT_LEVEL = {1:logging.DEBUG, 2:logging.INFO, 3:logging.WARNING, 4:logging.ERROR}
        
        Return(dict):
            return {'logger': logger, 'log_name': logname, 'logPth': logPth, 'tPth': tPth, 'writer': writer}
        """
        # ================================= Path process =========================================
        if prefix is None:
            prefix = time.strftime('%Y-%m-%d_%I-%M-%S_%p')
        # dir structure: log + (backbone+cls) + time + .log/tensorboard files
        # FIXME: make it default for all the class
        basicPth = log_dir
        basicPth = os.path.join(basicPth, 
                                self.model_t['cls_model'] + '_' +\
                                self.model_t['feat_model'] + '_' +\
                                self.dataset_opt['datatype'])
        basicPth = os.path.join(basicPth, prefix)
        # if the dir is not exist we create it 
        if not os.path.exists(basicPth):
            os.makedirs(basicPth)
        # save the log files here
        logPth = os.path.join(basicPth, logname + '.log')
        tPth = os.path.join(basicPth, 'tensor_writer')
        
        # ================================= logging part =========================================
        DEFAULT_LEVEL = {1:logging.DEBUG, 2:logging.INFO, 3:logging.WARNING, 4:logging.ERROR}
        # 初始化logger和tensorboard的writer
        logger = logging.getLogger(logname)
        logger.propagate = 0
        logger.setLevel(DEFAULT_LEVEL[console_level])

        # create file & console handler 
        fh = logging.FileHandler(logPth)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(DEFAULT_LEVEL[console_level])
        
        # create output format  for all the handler 
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s ', 
                    datefmt='%Y-%m-%d %I:%M:%S %p')

        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        # add handler to logger
        logger.addHandler(ch)
        logger.addHandler(fh)

        # record the hyper parameters into local log
        dataset_hyper = self.config.get('data')
        
        record_list = [dataset_hyper, self.training_opt]
        for record in record_list:
            if record is dataset_hyper:
                logger.info('DATASET: ' )
            else:
                logger.info('TRAINING_OPT ')
            if isinstance(record,dict):
                for k, v in record.items():
                    logger.info('{} : {}'.format(k, v))
        
        # ================================= init decorator =================================
        # FIXME: setting the right logger for each process and comfirm all the methods' usage
        self.save_model_v0.set_logger(logname)
        self.train.set_logger(logname)
        
        self.train_rec.update.set_logger(logname)
        self.train_rec.update.set_phase('training_')

        # ================================= writer part ===========================================
        # 如果已经有了指定的writter的指定名称，就按照指定的名称取创建或者覆写，如果没有的话就按照时间来指定相应的地址
        
        writer = SummaryWriter(tPth)
        
        # setting logger for training 
        self.train_rec.update.set_writer(writer)
        self.train_rec.final.set_writer(writer)
        
        # self.val_rec.update.set_writer(writer)
        dataVisualization.set_writer(writer)

        tensortext = ''
        for k,v in self.config.items():
            if isinstance(v,dict):
                tensortext += "{} \n".format(k)
                for k2,v2 in v.items():
                    tensortext += "{} : {}   \n".format(k2,v2)
            else:
                tensortext += "{} : {}   \n".format(k,v)
        # print(tensortext)
        writer.add_text('config', tensortext)

        # return name of loggger & logger itself
        return {'logger': logger, 'log_name': logname, 'logPth': logPth, 'tPth': tPth, 'writer': writer}

    def _freeze(self,model,freeze_type='all'):
        # all, feat, cls ...

        pass