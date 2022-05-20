"""
@AikenHong 2021
- Inherit the base runner to develop the final one
- Wrap the test function to support the whold process
    - Add the distill method in it
    - Add the Cluster and in it
    - Get the confi, logits, feature of test process

Pay attention to:
    when we using the owl_runner, which means there must be a keyword "new"

"""
# [ ] try SCL->SSL loss, update the scheduler for the loss setting

from tqdm import tqdm
import copy

import torch
from torch import nn, optim
from torch.utils import data

from data.setTransformer import select_transform
from data.datasetGe import *
from util.metric import *
from layers.ezConfidence import leastConfi
from layers.clusters import *
from util.utils import flatten_batch as f_b
# from util.runningScript import base_runner
from util.ssl_runner import SelfSupervisedTrainer
from torch.cuda.amp import GradScaler, autocast as autocast

from layers.warmup import GradualWarmupScheduler
from loss.custom_loss import EpoesCombiner, mixup_criterion

class owl_runner(SelfSupervisedTrainer):
    def __init__(self, config):
        super(owl_runner, self).__init__(config)
        # the super class will init the loger recorder and lists of attributes
        # we only need to check some result and add some specic operator here
        self.ocfg = config['online']
        self.update_times = 0
        self.clust_dict = self.ocfg['cluster_dict']
        self.down_dict = self.ocfg['downsample_dict']
        self.training_mode = self.config['training_mode']

        return None

    def main_process(self):
        # run pretrain -> main train -> online distill -> fi_test -> fi
        main_model, projector = self.entry_whold_ssl(mode=self.training_mode)

        if self.config['online']['enable']:
            isprojector = self.config['pretrain']['using_projector'] and \
                        self.config['pretrain']['ispretrain']
            if not isprojector: projector = None
            else: projector.eval()
            self.online_process(main_model, projector=projector)

        self.logger.info('Finish the All the process')

        return None

    def online_process(self, main_model, projector=None,
                        phase='', *args, **kwargs):
        """
        Desc:
        This function is used to get the online running process,
        which is a duplicate version of test.

        Wrap the test process to get new dataset(save and ),
        update base model(running distill proces).
        """
        # import os
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        """
         # 0 set up the memory space for the new,old data we selected,
        new_datas, new_features, new_labels = [], [], []

        # 1. running the model to collect those new class data (we only do the incremental once)
        # when we enable the collator, we donot do calculate the metric
        self.new_dataset.set_out_idx(True)

        try:
            self.new_dataset.transform = self.new_dataset.transform.base_transform
        except:
            pass

        new_dataloader = data.DataLoader(self.new_dataset, **self.dataopt)
        # which is depend on the paradigam is common or complex
        temp_freq = self.freq
        self.freq = None
        self.val_rec.final.set_verbose(False)
        _ = self.varify_v2(main_model, self.criterion, self.val_rec, new_dataloader,
                            phase='online_Collector_', collate_data=True, confi_thres=0.5, new_datas=new_datas,
                            new_labels=new_labels, new_features=new_features, *args, **kwargs)
        self.new_dataset.set_out_idx(False)
        
        new_idx = copy.deepcopy(new_datas)
        new_datas = torch.tensor(self.new_dataset.data)[new_idx,...]
        new_labels = torch.tensor(self.new_dataset.targets)[new_idx]
        # new_datas, new_features, new_labels = torch.cat(new_datas), \
        #             torch.cat(new_features), torch.cat(new_labels)
        new_features = torch.cat(new_features)
        
        self.val_rec.final.set_verbose(True)
        self.freq = temp_freq

        # 2. initialize the labelGE by the whole new label, and then we will update it in the training
        with torch.no_grad():
            new_features = torch.nn.AdaptiveAvgPool2d(1)(new_features).squeeze()
        labelGe = LabelGenerator('kmeans', self.clust_dict, 'pca', self.down_dict, projector=projector)
        labelGe.update(new_features)

        # 3. define new dataset, then we mix it with the old(include in the incremental)
        transform = select_transform(self.config['data']['transformer'], False)
        self.online_new_dataset = AllData(datatype='Given', datalist=f_b(new_datas),
                        targetlist=f_b(new_labels), transform=transform) 
        """
        labelGe=None
        # 4, then we running the new distill training. which should integrate generate new label in it
        #    so we should rewrite the train script for distill process
        self.train_rec.__init__()
        update_res = self.incremental(main_model, labelGe=labelGe, new_num_cls=None, hidden_layers=None,
                        dataset=self.dataset, new_dataset=self.new_dataset, *args, **kwargs)

        # 5. then we will finish the model update,
        # #    we need to align the true labels
        # new_features, new_labels = self.get_features(main_model, self.online_new_dataset)
        # labelGe.update(new_features,new_labels)

        #    after all, we running the test again to get the new res
        test_res = self.test(main_model, self.test_rec, self.test_dataloader,
                        'distill_','vis',mapdict=None, *args, **kwargs)

        # Fi
        return None

    @log_timethis(desc="distill training")
    def incremental(self, model, labelGe, new_num_cls=None, hidden_layers=None, dataset=None,
                    new_dataset=None, *args, **kwargs):
        """
        Suppose we have a trained model and a label generator.
        carry out this function after train or resume.

        Then we need to:

            1. use LabelGe to define a Loss calculator
            2. mix new(which have a special label) and old dataset to get a specific datagenerator
            3. then we tranfer it to the train epoch which can help us train the model.

        In the future:

            - using a new config to cover the old config.
            - we need to seprate the configuration like lr or optimizer of normal and distill

        """
        # in the future version, we may need to rewrite the trainbatch to support contrasitive
        # 0. init the mix dataset for it
        # lam = len(new_dataset) / len(dataset)
        lam = len(new_dataset) / len(dataset)
        transformer = select_transform('cifar100pil',True, i_size=32)
        augs = select_transform('simclr',True, i_size=32)
        # now we only support the balance random select
        from data.dataAugment import ContrastiveLearningViewGenerator
        transformer = ContrastiveLearningViewGenerator(
            transformer,
        )
        # if we want to using augmentation, we need to know the datatype of img right now
        dis_dataset = MixDataset(dataset, self.new_dataset, strategy='balance',
                                factor=lam, transformer=transformer, augs=augs)
        # ====================
        # from data.dataAugment import ContrastiveLearningViewGenerator
        # transformer = select_transform('cifar100',True, i_size=32)
        # transformer = ContrastiveLearningViewGenerator(
        #     transformer,
        # )
        # dataset = self.new_dataset
        # dataset.transform = transformer
        # dis_dataset = dataset
        # ====================
        dis_dataloader = data.DataLoader(dis_dataset, **self.dataopt)

        # 1. duplicate the model with the new_num_cls, reinit by default
        if new_num_cls is None:
            new_num_cls = self.model_t['num_cls'] + self.dataset_opt['num_new']
            self.model_t['num_cls'] = new_num_cls

        old_model = copy.deepcopy(model)
        model._expand_dim(new_num_cls, re_init=True, hidden_layers=hidden_layers, distill=True)
        new_model = model

        if torch.cuda.is_available():
            # fix the old model and we dynamically update the new model
            # may usign the EMA-like.
            old_model.cuda()
            new_model.cuda()
            old_model.eval()
            new_model.train()

        # 2.init the new optim, in the ditill we can change params here
        optim_ = self.init_optim(self.ocfg['optimizer'])
        bb_opt_dict = self.ocfg['optim_opt']
        clf_opt_dict = self.ocfg['clf_optim_opt']
        dis_optimizer = optim_(
            [{'params': new_model.backbone.parameters(), 'lr': bb_opt_dict['lr']},
            {'params': new_model.classifier.parameters(),}], **clf_opt_dict)

        # 3. init the warmup and scheduler for the optimizer
        schedule_t = self.ocfg['schedule']
        sche_dict = self.ocfg['sche_opt']
        schedule_f = self.init_schedule(schedule_t)

        after_schedule = schedule_f(dis_optimizer, **sche_dict)
        if self.ocfg['warmup']:
            dis_schedule = GradualWarmupScheduler(
                dis_optimizer,
                after_scheduler= after_schedule,
                **self.ocfg['warmup_opt']
            )

        # 4. Define our Loss Function here (sup_contrastive + kd + mixup-ce)
        # init the combiner of loss
        epoches = self.ocfg['epoches'] + 1
        self.combiner = EpoesCombiner(epoches=epoches-1, **self.ocfg['combiner_opt'])

        # init the basic loss
        self.criterion = self.init_loss(self.training_opt['loss'])
        # self.criterion = self.init_loss('arcface')
        
        # init the distill loss info 
        self.dis_criterion = self.init_loss(self.ocfg['dis_criterion'],**self.ocfg['dis_loss_opt']) # lam*criterion + (1-lam)*kd
        self.dis_criterion.criterion = self.criterion 

        # init the mix loss
        self.mixupCriterion = mixup_criterion(self.criterion)

        # init the contrast loss
        self.sup_contrastive = self.init_loss('supcontrast')

        if len(self.config['cuda_devices'].split(','))>1:
            new_model = nn.DataParallel(new_model)

        # 5. start ditsll training of it, and in here the val_dataloader should be the whole one
        # modify the save model when jthe phase is distill, we do not want it replace the origin model
        s_epoch = 0
        self.embed_mean = None
        if self.config['online']['resume']:
            resume = self.load_model_v0(new_model, dis_optimizer,
                                        self.config['online']['ckpt'])
            s_epoch = 151
        self.embed_mean = torch.zeros(int(self.config['networks']['in_dim'])).numpy()
        
        self.val_rec.__init__()
        self.train_epoch(epoches=epoches,
                         main_model=new_model,
                         optimizer=dis_optimizer,
                         criterion=self.criterion,
                         dataloader=dis_dataloader,
                         val_dataloader=self.test_dataloader,
                         new_trainer=self.train_dis_nbatch,
                         s_epoch=s_epoch,
                         schedule=dis_schedule,
                         phase='distill_',
                         val_n=1,
                         old_model=old_model,
                         labelGe=labelGe)

        # 6. save the final distill model for more usage.
        # self.save_model_v0(new_model, dis_optimizer,type='distill')
        return None

    def train_dis_nbatch(self, new_model, dataloader, criterion, optimizer, scalers, 
                    cur_epoch, train_rec, old_model, labelGe, idx_new=80,
                    *args, **kwargs):
        """
        This method will operate those complite training then basic distill.
        So we make it sepately.
        """
        # 1. set up the basic params include recorder before loop
        train_rec = self.train_rec if train_rec is None else train_rec
        max_mix_epoch = self.ocfg['mix_epoches']
        # & set up to calculate the moving avg of feature
        flat_feat = nn.AdaptiveAvgPool2d((1,1)).cuda()
        current_lr = min(1.0, optimizer.param_groups[0]['lr']*50)
        self.mu = 1.0 - (0.1) * current_lr

        # 2. start train iterations for one epoch.
        for i, batches in enumerate(tqdm(dataloader, desc='EPOCH: {}'.format(cur_epoch))):
            # get the original data which is duplicate for the scl
            images, labels = batches
            img_a, img_b = images[0], images[1]
            if torch.cuda.is_available():
                img_a, img_b = img_a.cuda(), img_b.cuda()
                labels = labels.cuda()

            # #  we intergrate with mixup in early eopches
            # if cur_epoch< max_mix_epoch:
            #     mix_datas, target_bf, target_af, lam = myAugmentation.mix_up(img_a, labels,1)

            # try for amp
            with autocast():
                feat_a, pred_a = new_model(img_a)
                feat_b, _ = new_model(img_b)
                o_feat_a, o_pread_a= old_model(img_a)

                # flaten the feature
                # ATTENTION: this part may should be control by the epoches
                feat_a, feat_b = flat_feat(feat_a).squeeze_(), flat_feat(feat_b).squeeze_()

                # then we need to calculate the true label for the new classes first
                # and we should know if this can fit the amp, if not, we just move it out
                # generate_labels_time_test = time.time()
                # labels = [
                #     labelGe(feat_a[i].float())
                #     if labels[i] >= idx_new else labels[i]
                #     for i in range(len(labels
                # ]
                # labels = torch.tensor(labels).cuda()
                # self.logger.info("it take {} for the label geneation ".format(time.time()-generate_labels_time_test))

                # calculate the scl loss
                features = torch.cat(
                    (feat_a.unsqueeze(1), feat_b.unsqueeze(1)), dim=1)

                if self.ocfg['combiner_opt']['strategy'] == 'abort':
                    scl = torch.tensor(0)
                else:
                    scl = self.sup_contrastive(features, labels)

                # calculate the mixup loss
                if cur_epoch< max_mix_epoch:
                    mix_datas, target_bf, target_af, lam = myAugmentation.mix_up(img_a.float(), labels.long(), 1)
                    _, pred_m = new_model(mix_datas)
                    mix_loss = self.mixupCriterion(pred_m, target_bf, target_af, lam)

                else: mix_loss = self.criterion(pred_a, labels)

            mix_loss = self.combiner(cur_epoch, scl, mix_loss)

            # calculate the kd+(mix)ce loss. donot need the not zero loss here.
            loss = self.dis_criterion(pred_a, o_pread_a, labels, criterion = self.criterion, basic_loss=mix_loss)

            # add the zero loss 
            pre_set_loss = self.ocfg['notzeroloss']
            if pre_set_loss is not None:
                loss = (loss - pre_set_loss).abs() + pre_set_loss

            # calculate the m`oving average of feature
            self.embed_mean = self.mu * self.embed_mean + \
                (1 - self.mu) * feat_a.detach().mean(dim=0).view(-1).cpu().numpy()

            optimizer.zero_grad()
            scalers.scale(loss).backward()
            scalers.step(optimizer)
            scalers.update()

            train_res_dict = self.Metrics_tool(labels, pred_a, self.metrics_type)

            # decorator will pass it to the tensorboard and .log
            verbose = True if (self.freq is not None and i % self.freq == 0) else False
            if verbose: train_rec.update.set_verbose(True)
            train_res_dict['confi'] = leastConfi(pred_a)
            train_rec.update(train_res_dict, loss = loss.item())
            if verbose: train_rec.update.set_verbose(False)

        torch.cuda.empty_cache()

        # And when we update the labelGE, we can calculate the confi by the mix data(old data's label)
        # Then we can use it to update or doing sth else.
        # align with the true lables
        # new_features, _ = self.get_features(new_model, self.online_new_dataset)
        # labelGe.update(new_features)

        # using EMA to update the old model's backbone and update the projector

        return train_rec

    def get_features(self, model, dataset):
        """
        using this model to get the features and labels from the newdataset to update the labelGE
        """
        new_features, new_labels = [],[]
        model.cuda()
        model.eval()
        flat_feat = nn.AdaptiveAvgPool2d((1,1))
        flat_feat.cuda()

        dataloader = data.DataLoader(dataset, **self.dataopt)
        with torch.no_grad():
            for i, (imgs,labels) in enumerate(tqdm(dataloader, desc='collate_feat&la')):
                if torch.cuda.is_available(): imgs = imgs.cuda()
                n_feature, n_pred = model(imgs)
                n_feature = flat_feat(n_feature).squeeze_()
                new_features.append(n_feature)
                new_labels.append(labels)

            new_features = torch.cat(new_features, dim=0)
            new_labels = torch.cat(new_labels, dim=0)
            # predict = torch.cat(new_labels, dim=)

        return new_features, new_labels
