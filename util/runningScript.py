""" @Aiken 2021 April
This File is Write to manage the Running Process:
including the Traing Testing Varifying.
"""
# !! 框架设计的部分，如果有嵌套的函数的话，改变传参的方式（内层的参数用字典传），甚至可以连嵌套的函数一起传进去
# Basic Module
import os
import copy

import torchvision
from tqdm import tqdm
import logging
# Torch module
import torch
from torch import nn,optim
from torch.utils import data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast as autocast
# Local module
from util.Visualize import tsne_visual
from util.utils import *
from util.wraps import *
from util.metric import Recorder,Metric_cal
from data.dataUtils import *
from data.datasetGe import AllData, DataSampler
from data.setTransformer import select_transform
from model import Assembler,M_select as MS
from loss import L_select as loss_setter
from layers.warmup import GradualWarmupScheduler
from layers.ezConfidence import leastConfi
# acceleate module
# try:
#     from apex import amp
#     from apex.parallel import DistributedDataParallel as DDP
#     from apex.parallel import convert_syncbn_model
# except ImportError:
#     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

# then write your own training process to override the train and test function
class base_runner():
    def __init__(self,config):
        super(base_runner, self).__init__()
        # loading based parameters & parser some important parameters
        self.config = config
        self.model_t = config['networks']
        
        self.val_n = config['val_n']
        self.freq = config['training_opt']['freq_out']
        self.resume = config['resume']
        # change the type of the data
        if isinstance(self.resume,str):
            self.resume = self.resume == 'True'
        
        # loading training opt dict for training process
        self.pretrain_opt = config['pretrain']['pretrain_opt']
        self.training_opt = config['training_opt']
        self.dataset_opt = config['data']['dataset']
        self.dataopt = self.training_opt['dataopt']
        self.drop = self.training_opt['train']['drop_rate'] 
        
        # init basic recorder for train and val 
        self.train_rec = Recorder()
        self.val_rec = Recorder()
        self.test_rec = Recorder()
        
        # set metric here or in the yaml files 
        # init metrics for train and val
        self.Metrics_tool = Metric_cal()
        self.metrics_type = config['metrics_type']

        # initialize the logger without parameters ,prefix
        logger_dict = self.init_logs(**config['logger'])
        self.logger = logger_dict['logger']
        self.writer = logger_dict['writer']
        self.t_pth = logger_dict['tPth']
        
        # init the projector in here, using this to do cluster in the online runner 
        self.projector = None

        # call the func to import apex module
        # self.apex_flag = try_import_apex(self.config['apex'])
        self.apex_flag = config['apex']
        self.START_EPOCH = 1
        self.fi_init_dataset = False

        # add some specific params here LT
        self.embed_mean = torch.zeros(int(config['networks']['in_dim'])).numpy()
        self.mu = 0.9
        
        return None
    
    #  ======================================= MAIN FUNCTION =========================================
    @log_timethis(desc = "training ")
    def train(self, pretrain_model=None, *args, **kwargs):
        """Init the training params, set up model loss optim etc."""
        # 0. init dataset
        self.init_dataset()
        
        # 1. init the basic model (consider pretrain or not.)
        self.main_model = self.init_model(pretrain_model=pretrain_model)
        
        # 2. init the optimizer
        optim_ = self.init_optim(self.training_opt['optimizer'])
        clf_opt_dict = self.training_opt['clf_optim_opt']
        bb_opt_dict = self.training_opt['optim_opt']
        # add pretrain method: diff lr setting
        if self.config['pretrain']['ispretrain'] and self.config['pretrain']['diff_lr']:
            self.optimizer = optim_([{'params': self.main_model.backbone.parameters(), 'lr': bb_opt_dict['lr']},
                                    {'params': self.main_model.classifier.parameters()}],
                                    **clf_opt_dict)
        else:
            self.optimizer = optim_(self.main_model.parameters(),**clf_opt_dict)

        # 3. init the schedule of lr in optimizer
        schedule_t = self.training_opt['schedule']
        schedule_f = self.init_schedule(schedule_t)
        sche_dict = self.training_opt['sche_opt'] 
        after_schedule = schedule_f(self.optimizer,**sche_dict)

        if self.training_opt['warmup'] == 'bbn':
            # this warmup is for the strong baseline.
            # we add it after that, so may see diff in the logs
            from layers.warmup import WarmupMultiStepLR
            self.schedule = WarmupMultiStepLR(
                self.optimizer,
                [120,160],
                0.01,
                warmup_epochs=5
            )
            
        elif self.training_opt['warmup']:
            self.schedule = GradualWarmupScheduler(
                self.optimizer,
                after_scheduler = after_schedule,
                **self.training_opt['warm_up']
            )
        
        else: self.schedule = after_schedule

        # 4. init the basic criterion
        self.criterion = self.init_loss(self.training_opt['loss'])
        if torch.cuda.is_available(): self.criterion = self.criterion.cuda()

        # 5. set up apex to speed up the training 
        # if self.apex_flag:
        #     self.main_model, self.optimizer = amp.initialize(self.main_model, self.optimizer, opt_level ="O1")
        
        # 6. set up the distributed training
        if len(self.config['cuda_devices'].split(',')) > 1:
            # if self.apex_flag:
            #     self.main_model = DDP(self.main_model,delay_allreduce=True)
            # else:
            self.main_model = nn.DataParallel(self.main_model)

        # 7. resume the stage of training and the model
        if self.resume:
            resumeDic = self.load_model_v0(self.main_model, self.optimizer,type=self.config['resume_type'])
        
        # 8. start training+val process
        epoches = self.training_opt['train']['num_epochs'] + 1

        self.train_epoch(epoches=epoches, main_model=self.main_model, optimizer=self.optimizer,
                criterion=self.criterion, dataloader=self.dataloader, val_dataloader=self.val_dataloader,
                schedule=self.schedule,s_epoch=self.START_EPOCH, val_n=self.val_n,
                train_rec=self.train_rec, val_rec=self.val_rec,)

        # 9. test on the test dataset get final result 
        self.test(self.main_model, self.test_rec, self.test_dataloader, '','vis')
        # self.test(self.main_model, self.test_rec, self.test_dataloader, '')

        # self.test('vis')
        
        # 10.close the logger and writter in the main function
        # if self.writer != None: self.writer.close()
        return self.main_model, self.projector
    
    # 在这一块定义recorder和调用相应的metric，同时在这里返回每个epoch的输出，然后用Recorder2 输出这一部分
    def train_epoch(self, epoches, main_model, optimizer, criterion, dataloader, new_trainer=None, val_dataloader=None,
                    schedule=None,s_epoch = 1, val_n = None, train_rec=None, val_rec=None, phase='train',
                    *args, **kwargs):
        # init those paramerters 
        es = EarlyStopping(patience=self.training_opt['train']['patience'],
                        verbose=True,logger=self.logger)
        acc1_h = None
        # get the default value for those recoders
        train_rec = self.train_rec if train_rec is None else train_rec
        val_rec = self.val_rec if val_rec is None else val_rec
        schedule = self.schedule if schedule is None else schedule
        val_dataloader = self.val_dataloader if val_dataloader is None else val_dataloader

        scalers = GradScaler()
        # start training
        for i in range(s_epoch,epoches):
            # set up here becuz the valling will change thi
            phase1 = phase
            train_rec.update.set_phase(phase1)
            main_model.train()

            if not new_trainer:
                self.train_nbatch(main_model, dataloader, criterion, optimizer, scalers,
                                cur_epoch=i, train_rec=train_rec)
            else:
                new_trainer(main_model, dataloader, criterion, optimizer, scalers=scalers,
                            cur_epoch=i, train_rec=train_rec, **kwargs)
            
            if self.config['pretrain']['ispretrain'] and self.config['pretrain']['diff_lr']:
                self.writer.add_scalars('Weights/lr',{'lr': optimizer.param_groups[0]['lr'],
                                                    'lr2': optimizer.param_groups[1]['lr']}, i)
            else:
                self.writer.add_scalars('Weights/lr',{'lr': \
                                    optimizer.param_groups[0]['lr']},i)
            phase2 = phase + '_epoch'
            train_rec.final.set_phase(phase2)
            train_rec.final()
            if val_n is None: schedule.step(train_rec.recorder['acc1'])
            train_rec.reset()
            
            # visualize those parameter as historgram
            # we can add other model here
            if i % 10 == 0:
                for name,param in main_model.named_parameters():
                    self.writer.add_histogram('main_model'+name,param.clone().cpu().data.numpy(),i)
                pass

            # save best model and using early stop
            if val_n is not None and i%val_n == 0:                
                # update the best result for save and es
                result_val = self.varify_v0(main_model, criterion, val_rec, val_dataloader,phase=phase)
                acc1_h,symbol_a = new_max_flag(acc1_h,result_val['acc1'])
                # [ ] we may add the conditional for this or using try except method
                schedule.step(metrics=result_val['acc1'])
                # schedule.step()
                if symbol_a:
                    subfix = self.t_pth.split('/')[-2]
                    self.save_model_v0(main_model, optimizer, i=i, subfix=subfix,log_path=self.t_pth)
                if es.step(result_val['acc1']):break
        
        self.logger.info('> END of TRAINING <')

        return None
 
    def train_nbatch(self, main_model, dataloader, criterion, optimizer, scalers,
                    cur_epoch=0, train_rec=None, **kwargs):
        # each epoch may run n batch，and we will upload info to tensorboard every batch
        Loss = 0
        train_rec = self.train_rec if train_rec is None else train_rec

        # training with the iterations setting, but we donot use this in normal situation
        """ for i in tqdm(range(iterations),desc='EPOCH: {} '.format(cur_epoch)):
        
            get data and label
            try:
                img,label = next(self.dataiterater)

            except StopIteration:
                self.logger.info('!!!run out of data ,reloader iterator')
                self.dataiterater = iter(self.dataloader)
                img,label = next(self.dataiterater) """
        flat_feat = nn.AdaptiveAvgPool2d((1,1)).cuda()
        current_lr = min(1.0, optimizer.param_groups[0]['lr']*50)
        self.mu = 1.0 - (0.1) * current_lr

        for i,datas in enumerate(tqdm(dataloader, desc='EPOCH: {} '.format(cur_epoch))):
            img,label = datas[0], datas[1]
            
            # set up cuda 
            if torch.cuda.is_available():
                label = label.cuda()
                img = img.cuda()
            
            with autocast():
            # using model to make prediction and conpare with labels
                if self.model_t['cls_model'] == 'Defaults' or self.model_t['feature_vis'] == False:
                    pred = main_model(img)
                else:
                    fa, pred = main_model(img)
                    fa = flat_feat(fa).squeeze_()

                Loss = criterion(pred,label)
                # using not zero loss to import the generalization of model
                pre_set_loss = self.training_opt['notzeroloss']
                if pre_set_loss is not None: 
                    Loss = (Loss - pre_set_loss).abs() + pre_set_loss
                
            # calculate the moving average of features
            self.embed_mean = self.mu * self.embed_mean + (1 - self.mu) * fa.detach().mean(dim=0).view(-1).cpu().numpy()

            optimizer.zero_grad()
            scalers.scale(Loss).backward()
            scalers.step(optimizer)
            scalers.update()

            # try:
            train_res_dict = self.Metrics_tool(label,pred,self.metrics_type)
            
            # except:
                # print('pred: {} \n label {} \n loss {}'.format(label, pred, Loss))
                # print('the shape of those parameters is {} \n {} \n {} \n'.format(label.shape,pred.shape,Loss.shape))
                # raise TypeError('fix this problem')

            # Then the decorator will pass it to tensorboard and .log
            
            verbose = True if (self.freq is not None and i % self.freq == 0) else False
            if verbose: train_rec.update.set_verbose(True)
            train_res_dict['confi'] = leastConfi(pred)
            train_rec.update(train_res_dict, loss = Loss.item())
            if verbose: train_rec.update.set_verbose(False)

        torch.cuda.empty_cache()        
        return train_rec

    def test(self, main_model, RecorderV, dataloader, phase='', 
                *args, **kwargs):
        """
        we need to seprate test with varify， becuz this method will contain many visulizPe work.
        and this part will do more online job in the future.
        """
        # reverse map (we map the labels to the cluster result we generate)
        reverse_map = kwargs.get('mapdict', None)
        # set up the logger for test
        RecorderV = self.test_rec if RecorderV is None else RecorderV        
        RecorderV.update.set_phase(phase + 'test_')
        dataloader_this = self.test_dataloader if dataloader is None else dataloader

        # those params to generate random batch for projector
        idxs = 0
        vis_idxs = torch.randperm(len(dataloader_this))[:5]
        vis_datas, vis_labels, vis_imgs = [], [], []
        pr_labels, pr_probs = [], []

        # set up the model
        main_model.eval()
        # self.logger.info("START TEST")

        # start testing the model
        with torch.no_grad():
            for img,label in tqdm(dataloader_this, desc='TESTING'):
                idxs += 1
                
                if reverse_map is not None:
                    label = torch.tensor([label[i] if label[i]<80 else reverse_map[label[i].item()]
                                             for i in range(len(label))])
                
                # get data and set cuda
                if torch.cuda.is_available():
                    label = label.cuda()
                    img = img.cuda()
                
                # using model to make prediction and compare with labels
                if self.model_t['cls_model'] == 'Defaults' or self.model_t['feature_vis'] == False:
                    pred = main_model(img)
                else:
                    feature, pred = main_model(img, embed=self.embed_mean)
                    
                    # set the embeeding infomaion for visulize
                    if idxs in vis_idxs:
                        vis_datas.append(feature)
                        vis_labels.append(label)
                        vis_imgs.append(img)
                
                # set up the each cal acc calculator
                pr_labels.append(label)
                pr_probs.append(pred)

                # maybe we can do this by the label update part.

                # evaluate the result and store it in recorder
                test_rec_dict = self.Metrics_tool(label,pred,self.metrics_type)
                confidence = leastConfi(pred)

                # add the confidence to recorder
                test_rec_dict['confi'] = confidence
                RecorderV.addExcept('confi')
                RecorderV.update(test_rec_dict)

            RecorderV.final.set_phase(phase + 'test_epoch')
            res = copy.deepcopy(RecorderV.final())
            RecorderV.reset()

        # visulize the embedding pr_curve acc_each
        if 'vis' in args and self.model_t['cls_model'] != 'Defaults':
            # add projectors here
            # concat those batches of data and label
            vis_datas = torch.cat(vis_datas,dim=0)
            vis_labels = torch.cat(vis_labels,dim=0)
            vis_imgs = torch.cat(vis_imgs,dim=0)

            # resize the tensor to a vector
            vis_datas = vis_datas.view(vis_datas.size(0),-1)
            self.writer.add_embedding(vis_datas, vis_labels, vis_imgs)

            # ===========================================================
            # add pr curve here.
            pr_labels = torch.cat(pr_labels,dim=0)
            pr_probs = torch.cat(pr_probs,dim=0)
            pr_probs = torch.stack([F.softmax(value,0) for value in pr_probs])
            
            # add the acc calculator here
            pr_res_value, pr_res_index = pr_probs.topk(1,1,True,True)
            pr_res_index.squeeze_(1)
            
            # the res of each cls
            fmt = "{:<9} | {:5.3%}  \t  \n"
            res_str = "{:<9} | {:5}  \t  \n".format('cls_name', 'acc1')
            
            # calculate the prediction acc for the origin class
            for global_step in range(self.model_t['num_cls']):
                tensor_labels = pr_labels == global_step
                tensor_probs = pr_probs[:,global_step]
                self.writer.add_pr_curve(phase+'prcurve_classes',tensor_labels,tensor_probs,global_step)
                # calculate the acc1 for each class
                tensor_res_index = pr_res_index == global_step
                res_acc1 = torch.mul(tensor_res_index, tensor_labels).float().sum()/tensor_labels.float().sum()
                res_str += fmt.format(global_step,res_acc1) 

            self.writer.add_text(phase + 'pr_class_acc',res_str)

            #  visulize the confi of each new_cls
            self.Metrics_tool.calculate_each_confi.set_title(phase + 'new_cls_confi')
            self.Metrics_tool.calculate_each_confi(self.model_t['num_cls'],pr_res_value,pr_labels)
            
            # visualize the feature cluster by tsne, like the projector
            tsne_visual.set_phase('tsne result ' + phase)
            tsne_visual(vis_datas,vis_labels)
            self.logger.info("finish writing the tsne visulization")
            
        return res
        
    def varify_v0(self, main_model, criterion, RecorderV, dataloader, phase='' ,
                collate_data=False, confi_thres=0.5, new_datas=[], new_labels=[],
                new_features=[], *args):
        """
        Varify the model we given in the vari dataset and return the loss and acc;
        Collate_data: Enable the collate_data, we'll use the config_thres to get data from new Classes.
        """
        # choose dataset from test and val set.
        RecorderV = self.val_rec if RecorderV is None else RecorderV
        RecorderV.update.set_phase(phase + 'val')

        # change model status 
        main_model.eval()

        # can imporve this status in config 
        new_idx = self.config['networks']['num_cls']
        new_num = 0

        # cls_base acc: 
        if 'distill' in phase:
            correct = [0. for i in range(100)]
            total = [0. for i in range(100)]

        # start testing the model
        with torch.no_grad():
            for i, (img, label) in enumerate(tqdm(dataloader)):
                # if collate_data: label, real_idx = label
                if torch.cuda.is_available(): 
                    label = label.cuda()
                    img = img.cuda()
                
                if self.model_t['cls_model'] == 'Defaults' or self.model_t['feature_vis'] == False:
                    pred = main_model(img)
                else:
                    # the embed will only been used when the clf is causal-like
                    feature,pred = main_model(img,embed=self.embed_mean)

                # calculate those metrics for test and valling
                Loss = criterion(pred,label) if not collate_data else -1
                val_rec_dict = self.Metrics_tool(label,pred,self.metrics_type)

                if not collate_data:
                    confidence = leastConfi(pred)
                else:
                    confidence, each_confi = leastConfi(pred, avg=False)
                    # collate those data as new one and change it in place
                    new_num += len([idx for idx in range(len(label)) if label[idx] >= new_idx])
                    idx = [idx for idx in range(len(each_confi)) if each_confi[idx] < confi_thres]
                    new_datas.append(img[idx,:])
                    new_labels.append(label[idx])
                    new_features.append(feature[idx,:])
                
                verbose = True if (self.freq is not None and i % self.freq == 0) else False
                if verbose: RecorderV.update.set_verbose(True)
                val_rec_dict['confi'] = confidence
                RecorderV.update(val_rec_dict, loss = Loss)
                if verbose: RecorderV.update.set_verbose(False)

                if 'distill' in phase:
                    # calculate the acc in the head and the tail 
                    dis_prediction = torch.argmax(pred,1)
                    correct_num = dis_prediction == label.float()
                    for idx in range(len(label)):
                        correct[label[idx]] += correct_num[idx].float().item()
                        total[label[idx]] += 1
        
        # then we calculate the acc in each class 
        # we change this part by the config, if wrong, u can test the value is right or not.
        if 'distill' in phase:
            tempn = self.config['data']['num_new']
            tempo = self.config['data']['num_cls'] - tempn
            self.logger.info("old {} class: {}  new {} class {}".format(
                tempo, sum(correct[:tempo])/sum(total[:tempo]), 
                tempn, sum(correct[tempo:])/sum(total[tempo:])
            ))

        if collate_data:
            # get the pr result of the selection process
            _ = self.Metrics_tool.confi_metric(new_labels, new_num, new_idx-1, True)

        RecorderV.final.set_phase(phase + 'valing_epoch')
        res = copy.deepcopy(RecorderV.final())
        
        RecorderV.reset()
        return res

    def varify_v2(self, main_model, criterion, RecorderV, dataloader, phase='' ,
                collate_data=False, confi_thres=0.5, new_datas=[], new_labels=[],
                new_features=[], *args):
        """
        Varify the model we given in the vari dataset and return the loss and acc;
        Collate_data: Enable the collate_data, we'll use the config_thres to get data from new Classes.
        """
        # choose dataset from test and val set.
        RecorderV = self.val_rec if RecorderV is None else RecorderV
        RecorderV.update.set_phase(phase + 'val')

        # change model status 
        main_model.eval()

        # can imporve this status in config 
        new_idx = self.config['networks']['num_cls']
        new_num = 0

        # cls_base acc: 
        if 'distill' in phase:
            correct = [0. for i in range(100)]
            total = [0. for i in range(100)]

        # start testing the model
        with torch.no_grad():
            for i, (real_idx, img, label) in enumerate(tqdm(dataloader)):
                # if collate_data: label, real_idx = label
                if torch.cuda.is_available(): 
                    label = label.cuda()
                    img = img.cuda()
                
                # the embed will only been used when the clf is causal-like
                feature,pred = main_model(img,embed=self.embed_mean)

                # calculate those metrics for test and valling
                Loss = criterion(pred,label) if not collate_data else -1
                val_rec_dict = self.Metrics_tool(label,pred,self.metrics_type)

                if not collate_data: confidence = leastConfi(pred)
                else: 
                    confidence, each_confi = leastConfi(pred, avg=False)

                    # collate those data as new one and change it in place
                    new_num += len([idx for idx in range(len(label)) if label[idx] >= new_idx])
                    idx = [idx for idx in range(len(each_confi)) if each_confi[idx] < confi_thres]
                    new_labels.append(label[idx])
                    new_features.append(feature[idx,...])
                    idx = real_idx[idx]
                    new_datas += idx
                
                verbose = True if (self.freq is not None and i % self.freq == 0) else False
                if verbose: RecorderV.update.set_verbose(True)
                val_rec_dict['confi'] = confidence
                RecorderV.update(val_rec_dict, loss = Loss)
                if verbose: RecorderV.update.set_verbose(False)

                if 'distill' in phase:
                    # calculate the acc in the head and the tail 
                    dis_prediction = torch.argmax(pred,1)
                    correct_num = dis_prediction == label.float()
                    for idx in range(len(label)):
                        correct[label[idx]] += correct_num[idx].float().item()
                        total[label[idx]] += 1
        
        # then we calculate the acc in each class 
        if 'distill' in phase:
            self.logger.info("old 80 class: {}  new 20 class {}".format(
                sum(correct[:80])/sum(total[:80]), sum(correct[80:])/sum(total[80:])
            ))

        if collate_data:
            # get the pr result of the selection process
            _ = self.Metrics_tool.confi_metric(new_labels, new_num, new_idx-1, True)

        RecorderV.final.set_phase(phase + 'valing_epoch')
        res = copy.deepcopy(RecorderV.final())
        
        RecorderV.reset()
        return res

    #  ======================================== BUILD A MODEL ========================================
    # 如果需要自定义class的比如model或者loss，就按照再__init__中写setup的方法去做
    # 否则的话直接像select_optim 那样的写法去写就行了
    
    def init_model(self, model_t=None, vis=True, pretrain_opt=None, 
                                    pretrain_model=None,*args, **kwargs):
        # get params 
        model_t = self.model_t if model_t is None else model_t
        ispre = self.config['pretrain']['ispretrain']
        pretrain_opt = self.pretrain_opt if pretrain_opt is None else pretrain_opt
        # !! moving this to the pretrain, or this will cause error in baserunner.
        if ispre and pretrain_opt['type'] == 'train': 
            # modify to a new classifier and save projector
            if pretrain_model is not None:
                model = pretrain_model
                model.feature_vis = self.model_t['feature_vis']
                self.projector = copy.deepcopy(model.classifier)
                model._expand_dim(model_t['num_cls'], reinit=True, hidden_layers=model_t['hidden_layers'])
            else:
                model = Assembler(**model_t)
        else:
            # reinit model and load model(bb) and projector
            model = Assembler(**model_t)
            if ispre and pretrain_opt['type'] == 'load':
                model, self.projector = self.load_model_pre(model)
                
        self.logger.info("the MODEL-{} have : {} PARAMETERS".format(
                        self.model_t['feat_model'],count_params(model)) )        
        # cuda
        if self.config['cuda_devices'] == '-1': return model
        if torch.cuda.is_available(): 
            model = model.cuda()
            self.projector = self.projector.cuda() if self.projector is not None else None

        # if we want to draw multi-models, we need to wrap two model in one strucutre
        if vis:
            img = torch.rand(4,3,224,224).cuda()
            self.writer.add_graph(model,img)

        return model

    def init_loss(self, LOSSTYPE, *args, **kwargs):
        # 选择损失函数进行初始化，在Train中进行损失函数的设置和对全局的损失变量进行赋值
        loss = loss_setter(LOSSTYPE, *args, **kwargs)
        self.logger.info("using {} LOSS<".format(LOSSTYPE))

        # 将损失转移到CUDA上
        if self.config['cuda_devices'] == '-1': return loss
        if torch.cuda.is_available():
            try:
                loss = loss.cuda()
            except AttributeError:
                print('the loss is not cuda-able {}'.format(type(loss)))
        
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
        elif schedule_t == 'auto':
            return optim.lr_scheduler.ReduceLROnPlateau
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
        elif optimizer_t == 'AdamW':
            optim_ = optim.AdamW
        else:
            raise NotImplementedError('please add this optim in the function: {}'.format(self.init_optim.__name__))

        return optim_

    #  ==================================== LOAD and SAVE ============================================
    def load_model_v0(self, model, optimizer=None, ckptpth=None, type='ckpt', subfix=None):
        """ load ckpt or pretrain model part, single model 
        Args:
            desc (str, optional): Defaults to 'ckpt'.
        """
        # assert optimizer is not None, 'optimizer must be defined first~!'
        # assert model is not None, 'model must be defined first~!'

        # ================================= PATH COMPLETE ==============================
        ckptpth = self.config.get('ckpt_pth') if ckptpth is None else ckptpth
        assert os.path.exists(ckptpth), 'check ur ckptpath, if u want resume,or load pretrain model'

        # ================================= load universe parameters ================================= 
        checkpoint = torch.load(ckptpth)
        # save_model_dict = {k:v for k,v in checkpoint['model'].items() if 'backbone' in k and 'fc' not in k}
        save_model_dict = {k:v for k,v in checkpoint['model'].items()}
        model_dict = model.state_dict()
        model_dict.update(save_model_dict)
        model.load_state_dict(model_dict)
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        try:
            if checkpoint['embed_mean'] is not None:
                self.embed_mean = checkpoint['embed_mean']
        except:
            pass

        # ================================= load and specific parameters ================================= 
        self.START_EPOCH = checkpoint['i'] if checkpoint.get('i') else 1
        
        if checkpoint.get('log_path'):
            self.log_path = checkpoint['log_path']
        
        # recalculate the index for the recorder
        self.train_rec.recorder['epoch'] = self.START_EPOCH 
        self.train_rec.recorder['itera'] = checkpoint["train_iter"]
        self.val_rec.recorder['epoch'] = int(self.START_EPOCH/self.val_n)
        self.val_rec.recorder['itera'] = checkpoint['val_iter']

        # reset all the decorator which need writer or log 
        self.writer = SummaryWriter(self.log_path)
        self.set_writer_for_decorator(self.writer)

        return checkpoint
    
    @log_timethis('savemodel')
    def save_model_v0(self, model, optimizer, save_pth=None, i=None, 
                            log_path=None,subfix=None, type='ckpt'):
        # ================================= setting ckpt to save =================================
        ckpt = {}
        ckpt['model'] = model.state_dict()
        ckpt['optimizer'] = optimizer.state_dict()
        # if self.apex_flag: ckpt['amp'] = amp.state_dict()

        if i is not None: ckpt['i'] = i 
        if log_path is not None: ckpt['log_path'] = log_path
        
        ckpt['train_iter'] = self.train_rec.recorder['itera']
        ckpt['val_iter'] = self.val_rec.recorder['itera']
        ckpt['embed_mean'] = None
        if self.config['training_mode'] == 'imb':
            ckpt['embed_mean'] = self.embed_mean
        
        # ================================= setting path ========================================
        now = time.strftime('%Y-%m-%d_%I-%M-%S_%p')
        # Update the way we save model and edit code in this part
        # !! add hyper params if we save once in next 1. version.
        if not subfix: subfix = now
        
        ckpt_pth = self.config.get('save_pth') if save_pth is None else save_pth
        ckpt_pth = os.path.join(ckpt_pth, type)
        ckpt_pth = os.path.join(ckpt_pth, 
                                self.model_t['cls_model'] + "_" +\
                                self.model_t['feat_model']) + "_"+\
                                self.dataset_opt['datatype']
        if not os.path.exists(ckpt_pth): 
            os.mkdir(ckpt_pth)
        ckpt_pth = os.path.join(ckpt_pth,subfix +'.pt')
        
        # ================================= saving it ============================================
        torch.save(ckpt, ckpt_pth)
        return ckpt

    @log_timethis('load_pretrain_model')
    def load_model_pre(self, model, projector=None, save_pth=None):
        # get path and load ckpt
        save_pth = self.pretrain_opt['pth'] if save_pth is None else save_pth
        assert save_pth is not None, 'you need to specify the save path in at least one place'
        ckpt = torch.load(save_pth)['model']
        
        # load base model
        model_dict = model.state_dict()
        pretrain_dict = {k:v for k,v in ckpt.items() if 'backbone' in k and 'fc' not in k}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)

        # load projector
        if projector is None:
            projector = MS(modelName=self.config['pretrain']['network']['cls_model'],
                                    **self.config['pretrain']['network'])
        projector_dict = projector.state_dict()
        pre_projector_dict = {k.replace('classifier.', ''):v for k,v in ckpt.items() if 'classifier' in k}
        projector_dict.update(pre_projector_dict)
        try:
            projector.load_state_dict(projector_dict)
        except:
            print("failure to load projector")

        return model, projector

    #  ======================================= utils and initialize  =================================
    def init_logs(self, logname = 'TrainLoger', log_dir='./log', console_level = 2, prefix = None, subfix = None):
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
        basicPth = log_dir
        dirname = ''
        if subfix is not None:
            dirname += subfix
        dirname += self.model_t['cls_model'] + '_' + \
                    self.model_t['feat_model'] + '_' +\
                    self.dataset_opt['datatype']
        basicPth = os.path.join(basicPth, dirname)
        basicPth = os.path.join(basicPth, prefix)
        # if the dir is not exist we create it 
        if not os.path.exists(basicPth):
            os.makedirs(basicPth)
        # save the log files here
        logPth = os.path.join(basicPth, logname + '.log')
        tPth = os.path.join(basicPth, 'tensor_writer')

        # save config to the new yaml in the same place
        import yaml
        with open(os.path.join(basicPth, 'config.yaml'), "w") as f:
            yaml.dump(self.config, f)
        
        # ================================= logging part =========================================
        DEFAULT_LEVEL = {1:logging.DEBUG, 2:logging.INFO, 3:logging.WARNING, 4:logging.ERROR}
        # 初始化logger和tensorboard的writer
        logger = logging.getLogger(logname)
        logger.propagate = 0
        logger.setLevel(logging.DEBUG)

        # create file & console handler 
        ch = logging.StreamHandler()
        ch.setLevel(DEFAULT_LEVEL[console_level])
        fh = logging.FileHandler(logPth)
        fh.setLevel(logging.DEBUG)

        
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
        
        record_list = [dataset_hyper, self.training_opt, self.model_t]
        for record in record_list:
            if record is dataset_hyper:
                logger.info('DATASET: ' )
            elif record is self.training_opt:
                logger.info('TRAINING_OPT ')
            if isinstance(record,dict):
                for k, v in record.items():
                    logger.info('{} : {}'.format(k, v))
        
        # ================================= init decorator =================================
        self.save_model_v0.set_logger(logname)
        self.load_model_pre.set_logger(logname)
        self.train.set_logger(logname)
        
        self.train_rec.update.set_logger(logname)
        self.train_rec.final.set_logger(logname)
        # ================================= writer part ===========================================
        # 如果已经有了指定的writter的指定名称，就按照指定的名称取创建或者覆写，如果没有的话就按照时间来指定相应的地址
        
        writer = SummaryWriter(tPth)
        
        # setting logger for training 
        self.train_rec.update.set_writer(writer)
        self.train_rec.final.set_writer(writer)
        
        # self.val_rec.update.set_writer(writer)
        self.Metrics_tool.calculate_each_confi.set_writer(writer)
        dataVisualization.set_writer(writer)
        tsne_visual.set_writer(writer)
        self.Metrics_tool.confi_metric.set_writer(writer)

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

    def init_dataset(self,**kwargs):
        if self.fi_init_dataset == True: return None
        else: self.fi_init_dataset = True

        # init the dataset
        transformer = select_transform(self.config['data']['transformer'], True)
        if kwargs.get('warpaugs'):
            from data.dataAugment import ContrastiveLearningViewGenerator
            transformer = ContrastiveLearningViewGenerator(
                transformer,
            )
        dataset = AllData(**self.dataset_opt, transform=transformer)
        
        self.dataset_opt['istrain'] = 1
        transformer = select_transform(self.config['data']['transformer'], False)
        test_dataset = AllData(**self.dataset_opt)
        
        self.dataset_opt['istrain'] = 2
        transformer = select_transform(self.config['data']['transformer'], False)
        val_dataset = AllData(**self.dataset_opt)
        
        # samplig the dataset by diff strategy
        data_tag = self.config['data']['preprocess']
        if data_tag:
            TRAIN_DATASET = DataSampler(dataset, **self.dataset_opt)
            if data_tag == 'imb': 
                dataset = TRAIN_DATASET.get_imb()
            
            elif data_tag == 'new':
                dataset = TRAIN_DATASET.get_remain()
                new_dataset = TRAIN_DATASET.new_dataset
                self.new_dataset = new_dataset

                VAL_DATASET_RE = DataSampler(val_dataset, **self.dataset_opt)
                val_dataset = VAL_DATASET_RE.get_remain()

                # generate the label mapping for the test dataset
                temp_v = self.dataset_opt['num_new']
                self.dataset_opt['num_new'] = 0
                TEST_DATASET_RE = DataSampler(test_dataset, **self.dataset_opt)
                test_dataset = TEST_DATASET_RE.get_remain()
                self.dataset_opt['num_new'] = temp_v
            
            elif data_tag == 'both':
                # [ ] check that if we want to load model for distill, we need to fix the new-classes
                new_order = TRAIN_DATASET.process()
                self.dataset_opt['new_order'] = new_order
                dataset = TRAIN_DATASET.remain_dataset
                new_dataset = TRAIN_DATASET.new_dataset
                self.new_dataset = new_dataset

                # we need to comfirm the dnew_dataset's classes is the same
                VAL_DATASET_RE = DataSampler(val_dataset, **self.dataset_opt)
                val_dataset = VAL_DATASET_RE.get_remain()

                # mapping the label with the label in the train and val
                temp_v = self.dataset_opt['num_new']
                self.dataset_opt['num_new'] = 0
                TEST_DATASET_RE = DataSampler(test_dataset, **self.dataset_opt)
                test_dataset = TEST_DATASET_RE.get_remain()
                self.dataset_opt['num_new'] = temp_v
        
        # register the dataset into the class, 
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # and make it dataloader  
        self.dataloader = data.DataLoader(dataset, **self.dataopt)
        self.test_dataloader = data.DataLoader(test_dataset, **self.dataopt)
        self.val_dataloader = data.DataLoader(val_dataset, **self.dataopt)
        
        # self.new_dataset._save_origin_dataset(datasetname='cifar100_ow_new')
        # self.dataset._save_origin_dataset(datasetname='cifar100_ow_base')
        #self.val_dataset._save_origin_dataset(datasetname='cifar100_ow_val')
        #self.test_dataset._save_origin_dataset(datasetname='cifar100_ow_test')
         
        # display the imgs for the first time
        dataVisualization.set_phase('show dataloader once')
        samplingForDisplay(self.dataloader, sampleType=2)
        
        return None
    
    def set_writer_for_decorator(self, writer):
        """set up writer for all the decorator in this function"""
        self.train_rec.update.set_writer(writer)
        self.train_rec.final.set_writer(writer)
        dataVisualization.set_writer(writer)
        tsne_visual.set_writer(writer)
        self.Metrics_tool.confi_metric.set_writer(writer)

        return None

    def _freeze(self,model,freeze_type='all'):
        # all, feat, cls ...
        pass
