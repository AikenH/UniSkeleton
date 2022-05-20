"""
@AikenHong
This runner is designed for the Longtailed Situation.
Like Disrunner, we may try to reuse the train epoches.
Rewrite the Init of train and the iterations.

@reference:

"""
import torch
from torch import nn
from tqdm import tqdm
from util.wraps import *
from util.runningScript import base_runner
from data.dataAugment import myAugmentation
from layers.warmup import GradualWarmupScheduler
from loss.custom_loss import EpoesCombiner,mixup_criterion
from torch.cuda.amp import GradScaler, autocast as autocast

class ImbRunner(base_runner):
    def __init__(self, configs):
        super(ImbRunner, self).__init__(configs)
        # which make it in the base runner
        ...

    @log_timethis(desc='imbtrain')
    def imb_train(self, pretrain_model=None, *args, **kwargs):
        """
        if we overwrite the training, in the ssl part
        we will only suit for one situation which is what we donot want
        so we recreate a runner, if this if better than train, we overwrite it.
        """
        # 1. init the dataset
        self.init_dataset(warpaugs=True)

        # 2. init the basic model with imb config
        main_model = self.init_model(pretrain_model=pretrain_model)

        # 3. init the optimizer for backbone and classifier
        optim_ = self.init_optim((self.training_opt['optimizer']))
        clf_opt_dict = self.training_opt['clf_optim_opt']
        bb_opt_dict = self.training_opt['optim_opt']

        # if we want to frozon some parameters, we just _freeze and not pass in or lr=0
        if self.config['pretrain']['ispretrain'] and self.config['pretrain']['diff_lr']:
            optimizer = optim_(
                [{'params': main_model.backbone.parameters(), 'lr': bb_opt_dict['lr']}, 
                {'params': main_model.classifier.parameters()}], **clf_opt_dict)
        else:
            optimizer = optim_(main_model.parameters(),
                                    **clf_opt_dict)

        # 4. init the warmup and scheduler for the optimizer
        sche_dict = self.training_opt['sche_opt']
        schedule_t = self.training_opt['schedule']
        schedule_f = self.init_schedule(schedule_t)

        after_schedule = schedule_f(optimizer, **sche_dict)
        if self.training_opt['warmup']:
            schedule = GradualWarmupScheduler(
                optimizer,
                after_scheduler=after_schedule,
                **self.training_opt['warm_up'])
        else: schedule = after_schedule

        # 5. init the basic criterion causal: (sup_contrastive + mixup-ce/ce) disalign: (sup_contrastive + alignloss/ce)
        epoches = self.training_opt['train']['num_epochs'] + 1
        self.combiner = EpoesCombiner(
            epoches=epoches-1, **self.training_opt['combiner_opt'])

        self.criterion = self.init_loss((self.training_opt['loss']))
        self.mixupCriterion = mixup_criterion(self.criterion)
        self.sup_contrastive = self.init_loss('supcontrast')

        # 6. setup the distribute training
        if len(self.config['cuda_devices'].split(','))>1:
            main_model = nn.DataParallel(main_model)

        # 7. resume the stage of training and model
        s_epoch = 1 
        if self.resume:
            self.load_model_v0(main_model,
                               optimizer,
                               type=self.config['resume_type'])
            s_epoch = self.START_EPOCH
        
        # 8. start training+val process
        self.train_epoch(epoches=epoches,
                         main_model=main_model,
                         optimizer=optimizer,
                         criterion=self.criterion,
                         dataloader=self.dataloader,
                         new_trainer=self.train_imb_nbatch,
                         schedule=schedule,
                         s_epoch=s_epoch,
                         val_n=self.val_n,
                         phase='imb_training')

        # 9. test on the test dataset get final result
        self.test(main_model, self.test_rec, self.test_dataloader, '','vis')

        return main_model, self.projector

    def train_imb_nbatch(self, main_model, dataloader, criterion, optimizer,  scalers,
                        train_rec=None, cur_epoch=0, *args, **kwargs):
        """
        main body of the imb-training process, 
        we use train_epoch to call this function
        """

        train_rec = self.train_rec if train_rec is None else train_rec
        max_mix_epoch = self.training_opt['max_mix_epoch']
        # calculate the moving avg
        flat_feat = nn.AdaptiveAvgPool2d((1,1)).cuda()
        current_lr = min(1.0, optimizer.param_groups[0]['lr']*50)
        self.mu = 1.0 - (0.1) * current_lr

        for i, batches in enumerate(tqdm(dataloader, desc='EPOCH: {} '.format(cur_epoch))):
            images, labels = batches
            img_a, img_b = images[0], images[1]
            if torch.cuda.is_available():
                img_a, img_b = img_a.cuda(), img_b.cuda()
                labels = labels.cuda()

            # mixup data
            if cur_epoch< max_mix_epoch:
                mix_datas, target_bf, target_af, lam = myAugmentation.mix_up(img_a,labels,1)

            with autocast():
                # forward, using imga as the main data.
                fa, pred_a = main_model(img_a)
                fb, _ = main_model(img_b)

                # calculate loss and combine it which should be controled by epoch
                # we donot add projector head for our supcontrast.
                # supcon loss
                fa, fb = flat_feat(fa).squeeze_(), flat_feat(fb).squeeze_()
                features = torch.cat((fa.unsqueeze(1), fb.unsqueeze(1)), dim=1)
                scl = self.sup_contrastive(features, labels)

                # mixup loss in early epoch and ce loss in later epoch
                if cur_epoch< max_mix_epoch:
                    _, pred_m = main_model(mix_datas)
                    mixloss = self.mixupCriterion(pred_m, target_bf, target_af, lam)
                else:
                    mixloss = self.criterion(pred_a, labels)

                # combine the 
                loss = self.combiner(cur_epoch, scl, mixloss)
                # loss = mixloss + scl

                # not zero loss
                pre_set_loss = self.training_opt['notzeroloss']
                if pre_set_loss is not None:
                    loss = (loss - pre_set_loss).abs() + pre_set_loss

            # calculate the moving average of features
            self.embed_mean = self.mu * self.embed_mean + (1 - self.mu) * fa.detach().mean(dim=0).view(-1).cpu().numpy()

            # backward and update
            optimizer.zero_grad()
            scalers.scale(loss).backward()
            scalers.step(optimizer)
            scalers.update()

            # we can calculate acc according to the main stream of image
            # which is not diff with the normal situation
            train_res_dict = self.Metrics_tool(labels,pred_a,self.metrics_type)
            verbose = True if (self.freq is not None and i % self.freq == 0) else False
            if verbose: train_rec.update.set_verbose(True)
            train_rec.update(train_res_dict, loss = loss.item())
            if verbose: train_rec.update.set_verbose(False)

        torch.cuda.empty_cache()
        return train_rec
