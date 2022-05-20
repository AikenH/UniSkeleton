"""@AikenHong 2021 
Update configs with the commandline args
1. setting parameters by the config file
2. update it with the args which save those params will be update often
"""
import yaml
import argparse
 
def argparser():
    """[args(需要经常改变的参数) input and load config files（模型或者说是类别通用的参数）]

    Returns:
        [config]: [dict:]
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--cfg', default = r'config\mini_imagenet_resnet.yaml', type = str, help = "load configuration")
    parser.add_argument('--seed',default = None, type = int,
                         help = "using random seed to make experiments reproducible") 
    # parser.add_argument('--model_dir',default = None, type = str, help = "where to load model ")
    # train还是改成int把，代表val，train，test三种不同的阶段去做
    parser.add_argument('--train',default = 0, type = int, help = "0:train; 1:val; 2:test")
    parser.add_argument('--val_n',default = None, type=int, help= 'how much epoch we carry out 1 verify' )
    
    # 通过中间特征的存储，这一部分的特征存储是为了stage-two的使用，或者说是聚类可能性的分析，（sometimes it will help）
    parser.add_argument('--save_features',default = False, action = 'store_true',
                        help = "help to analysis the midden result")
    parser.add_argument('--cuda_devices', default = None, type = str, help = "using how man")
    parser.add_argument('--lr',default = None,type = float,help = 'learning rate')
    parser.add_argument('--batch_size',default = None, type = int)
    # parser.add_argument('--need_sampler',default = False, action = 'store_true',\
    #                     help = "when we using a complete dataset and want to imbalance")
    
    # Log设定，后续读取的时候自动变换为大写，如果不存在默认的几种，就使用默认的参数
    # parser.add_argument('--ch_logL',default = 'DEBUG', type = str, help = 'console level')
    # parser.add_argument('--fh_logL',default = 'WARNING', type = str, help = 'log file level')

    # Ckpt相关参数:存储模型的地址应该在配置文件中，迭代次数啊啥的就存在ckpt文件中就好
    parser.add_argument('--resume', default = None, help = 'load model or ckpt' )
    parser.add_argument('--resume_type',default = 'ckpt',type = str, help = 'is optional if u want to describe or not is fine')
    parser.add_argument('--ckpt_pth',default = None, type = str, help = "where to save and load ckpy ")
    
    args = parser.parse_args()

    # we can using var make args dict 
    # args = vars(args)

    # LOAD CONFIGURATION here if isnot none
    if args.cfg:
        print('> using {} configuration <'.format(args.cfg))
        with open(args.cfg) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # update config with args 
        updata(config, args)
        return config
    else:
        return vars(args)


def updata(config, args):   
    """ intergrate config and args, with the principle you want

    Args:
        config: [yaml config file load from hard drive]
        args
    """
    # UPDATE SPECIFIC KEY OF CONFIG 
    # config['model_dir'] = getValue(config['model_dir'], args.model_dir)
    config['training_opt']['dataopt']['batch_size'] = getValue(config['training_opt']['dataopt']['batch_size'], args.batch_size)
    config['training_opt']['optim_opt']['lr'] = getValue(config['training_opt']['optim_opt']['lr'], args.lr)
    
    ExceptKey = ['lr', 'model_dir', 'batch_size',]
    # UPDATA　ALL THE (REST) PARAMS ON CONFIG
    for key,value in vars(args).items():
        if key not in ExceptKey:
            if value is None: continue 
            config[key] = value
    
    if config['resume'] == True:
        config['pretrain']['ispretrain'] = False
        
    return config
    
    
def getValue(config_value,args_value):
    if args_value is not None:
        return args_value
    else:
        return config_value
