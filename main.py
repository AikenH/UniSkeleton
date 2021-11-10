"""@Aiken 2021 4/21 Entrypoint
load configuration and seting up the experimental variables
"""
# import basic libs 
import os
import random
# import torch libs 
import torch
# import files 
from config.argparser import argparser

if __name__ == "__main__":
    print(__doc__)
    
    # LOAD CONFIGURATIONs
    config = argparser()
    
    # SETTING GPU DEVICES  and random seed 
    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_devices']
    torch.backends.cudnn.benchmark = True

    # SETTING All Random seed
    if config.get('seed'):
        print("> Using Fixed Random Seed: {} <".format(config['seed']))
        
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        
        torch.cuda.manual_seed(config['seed'])
    
    # SETTING the CPU stat`us
    # from multiprocessing import cpu_count

    # cpu_num = cpu_count()
    # cpu_num = int(cpu_num/2)
    # os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    # os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    # os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    # os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    # os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    # torch.set_num_threads(cpu_num)

    from util.runningScript import base_runner
    # Main Processing which is along with the test process
    mainProcess = base_runner(config)
    mainProcess.train()

    print("End of all the process")

   

    
    