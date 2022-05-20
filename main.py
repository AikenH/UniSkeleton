"""
@Aiken: 2021 4/21 Entrypoint
@Desc: ->START YOUR PIPELINE<-
load configuration and seting up the experimental variables
setup the basic cpu and cuda status for training
"""
import os
import random
import torch
from config.argparser import argparser

if __name__ == "__main__":
    print(__doc__)
    config = argparser()
    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_devices']
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '5678'
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # torch.cuda.set_device(config['local_rank'])
        # torch.distributed.init_process_group(backend='nccl',
        #                                     init_method='env://',
        #                                     ) # 优先级
    if config.get('seed'):
        print("Using Fixed Random Seed: {}".format(config['seed']))
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
    
    # from multiprocessing import cpu_count
    # cpu_num = cpu_count()
    # cpu_num = int(cpu_num/2)
    # os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    # os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    # os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    # os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    # os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    # torch.set_num_threads(cpu_num)
    # ============================Online=============================================
    from util.dis_runner import owl_runner
    mainProcess = owl_runner(config)
    mainProcess.main_process()
    if mainProcess.writer is not None: mainProcess.writer.close()
    print("End of all the process")

   

    
    