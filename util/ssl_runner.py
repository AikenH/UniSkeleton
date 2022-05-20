"""
@AikenHong
Using This Module to Devlop SimCLR, After Coding 
Decides if we add it in the main runningscript

@reference:
1. https://github.com/lightly-ai/lightly
2. https://github.com/sthalles/SimCLR
"""
from tqdm import tqdm
from torch import nn
from torch.utils import data
from torch.cuda.amp import GradScaler, autocast as autocast
from data.dataAugment import ContrastiveLearningViewGenerator
from data.datasetGe import AllData
from data.setTransformer import simclr_transformer
from util.metric import *
from util.runningScript import base_runner
from util.lt_runner import ImbRunner

class SelfSupervisedTrainer(ImbRunner):
    def __init__(self, configs):
        super(SelfSupervisedTrainer, self).__init__(configs)
        # we will add pretrain in the configs
        self.ispretrain = configs['pretrain']['ispretrain']
        self.pretype = configs['pretrain']['pretrain_opt']['type']
        return None

    def entry_whold_ssl(self,mode='df'):
        """
        entry of the train process, if ispretrain==False, we will not run the ssl
        """
        pre_model = None
        # if it's the stage train, we will train ssl fitst
        if self.ispretrain and self.pretype == 'train':
            pre_model = self.pre_process()

        # if it's load, the process will load ssl model in the init model part
        if mode == 'df':
            main_model, projector = self.train(pretrain_model=pre_model)
        else:
            main_model, projector = self.imb_train(pretrain_model=pre_model)
            
        return main_model, projector

    @log_timethis(desc='self supervised pre-training')
    def pre_process(self):
        """
        Define the pre-training process, using a dataset without the label
        Define specific loss, optimizer and the model(with projector)
        首先写好simclr的训练过程，然后通过参数修正等方式，将其嵌入我们的框架中
        最主要的包括进行参数的共享，模型的共享，数据的混合等等
        """
        self.logger.info("SETTING PRE-TRAINING")
        # 0. init the dataset,
        pre_dataset_opt = self.config['pretrain']['dataset']
        #    using this to duplicate the image*2
        pre_transform = ContrastiveLearningViewGenerator(
            simclr_transformer,
        )
        if self.config['pretrain']['datasample'] == 'all':
            pre_dataset_opt['transform'] = pre_transform
            pre_dataset = AllData(**pre_dataset_opt)
        elif self.config['pretrain']['datasample'] == 'imb':
            self.init_dataset()
            temp_transform = self.dataset.transform
            pre_dataset = self.dataset
            pre_dataset.transform = pre_transform

        # 1. init the basic model with backbone + projector
        pre_model = self.init_model(self.config['pretrain']['network'])

        # 2. init the optimizer and schedule
        optim_ = self.init_optim(self.config['pretrain']['optim'])
        opt_dict = self.config['pretrain']['optim_opt']
        pre_optimizer = optim_(pre_model.parameters(), **opt_dict)

        # 3. init the schedule
        schedule_f = self.init_schedule('auto')
        pre_schedule = schedule_f(pre_optimizer, **self.config['pretrain']['sche_opt'])

        # 4. init the basic loss
        print(self.config['pretrain']['loss_opt'])
        pre_criterion = self.init_loss(**self.config['pretrain']['loss_opt'])
        if torch.cuda.is_available(): pre_criterion = pre_criterion.cuda()

        # 5. set up the distributed training
        if len(self.config['cuda_devices'].split(',')) > 1:
            pre_model = nn.DataParallel(pre_model)

        # 6. start the training(do not enable the val)
        epoch = self.config['pretrain']['epoch']
        self.logger.info('START PRE-TRAINING')
        tmp_model = self.pretrain(pre_model, pre_optimizer, pre_schedule, pre_criterion, pre_dataset, epoch)

        # 7. visual the feature
        self.feature_test(tmp_model, self.test_dataset)

        # 8 reinit the recorder we used in the ssl process
        self.train_rec.__init__()

        # 9 resume the dataset
        if self.config['pretrain']['datasample'] == 'imb':
            self.dataset.transform = temp_transform

        return pre_model

    def pretrain(self, model, optimizer, scheduler, criterion, dataset, epoch,
                    phase='pre_train', train_rec=None, *args, **kwargs):
        # resume the training style and model
        start_epoch = 0
        if False:
            ckpt = self.load_model_v0(model, optimizer, r'save_model/pre_train/mlp_cifar_rs50_cifar100/1024_81.pt')
            start_epoch = ckpt['i']

        # Analysis the diff from the basic training
        TAGS = time.strftime('%Y-%m-%d_%I-%M-%S_%p')
        train_rec = self.train_rec if train_rec is None else train_rec
        es = EarlyStopping(patience=self.training_opt['train']['patience'],
                            verbose=True,logger=self.logger)
        pre_dataloader = data.DataLoader(dataset, **self.config['pretrain']['dataload_opt'])

        scaler = GradScaler()
        for epoch_count in range(start_epoch, epoch):
            phase1 = phase+'_'
            train_rec.update.set_phase(phase1)
            model.train()

            for index, batches in enumerate(tqdm(pre_dataloader, desc='EPOCH: {} '.format(epoch_count))):
                images, _ = batches
                img_a, img_b = images[0], images[1]
                if torch.cuda.is_available():
                    img_a, img_b = img_a.cuda(), img_b.cuda()

                with autocast():
                    _, out1 = model(img_a)
                    _, out2 = model(img_b)
                    loss, logits, labels = criterion(out1, out2)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                # loss.backward()

                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()

                verbose = True if (self.freq is not None and index % self.freq == 0) else False
                if verbose: train_rec.update.set_verbose(True)
                res = self.Metrics_tool(labels, logits, self.metrics_type)
                train_rec.update(res, loss=loss.item())
                if verbose: train_rec.update.set_verbose(False)

            self.writer.add_scalars('Weights/lr',{'lr': optimizer.param_groups[0]['lr']},epoch_count)
            # get the metric and print it
            phase2 = phase + '_epoch'
            train_rec.final.set_phase(phase2)
            train_rec.final()
            scheduler.step(train_rec.recorder['acc1'])

            if es.step(train_rec.recorder['acc1']):
                self.logger.info('Early Stopping')
                break
            if es.flag:
                self.save_model_v0(model, optimizer, i=epoch_count,
                                    subfix=TAGS, type='pre_train', log_path=self.t_pth)
            train_rec.reset()

        self.logger.info('PRE-TRAINING fi')

        # we can pass the model or the save path, but we donot load the training params.
        # it depends
        return model

    def feature_test(self, model, dataset, num_need=10, phase='ssl'):
        """
        Attention: we only collate those features which in several class.
                and we should setup a uplimit for the num of features.

        And we should consider make this process as our basic Function.
        Becus we may use it for several times
        """
        # define collator and parameters
        collator = {
            'features': [],
            'labels': [],
        }
        num_class = 100
        classes_needed = torch.randperm(len(range(num_class)))[:num_need]

        # define model
        flattor = nn.AdaptiveAvgPool2d((1, 1))
        if torch.cuda.is_available():
            model.cuda()
            flattor.cuda()
        model.eval()

        # define dataloader
        dataloader = data.DataLoader(dataset, **self.dataopt)

        # running collation
        with torch.no_grad():
            for i, (imgs,labels) in enumerate(tqdm(dataloader,desc="collate_feats")):
                if torch.cuda.is_available(): imgs = imgs.cuda()
                vis_f,_ = model(imgs)
                feats = flattor(vis_f).squeeze_().detach()
                label = labels.cpu().numpy()

                # select those features which in the classes_needed
                feats = feats[np.isin(label, classes_needed)]
                label = label[np.isin(label, classes_needed)]
                collator['features'].append(feats)
                collator['labels'].append(label)

            # stack to get the right shape
            collator['features'] = torch.cat(collator['features'])
            collator['labels'] = np.hstack(collator['labels'])

        # pass features to the tsne
        from util.Visualize import tsne_visual
        tsne_visual.set_phase('tsne result' + phase)
        tsne_visual(collator['features'], collator['labels'], classes_needed.cpu().numpy())

        return None