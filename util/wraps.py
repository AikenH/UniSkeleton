"""
@Author: AikenHong 2021
@Desc: Move Wrapper To this file 
"""
import time
import logging
from functools import wraps, partial


""" =======DECORATOR: help to print some function description or sth. like this==== """

# 将figure排除出mode，独立出来

def attach_wrapper(obj, func = None):
    if func is None:
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)
    return func

#  以logging形式传入参数的装饰器，
def log_timethis(desc = 'train',log_name = 'train'):
    """通过logger的形式记录每段/监控程序的运行时间

    Args:
        desc (str): [用来帮助识别运行状态是训练还是测试等等]. Defaults to 'train'.
        log_name (str): [再外部定义好的logger的名称]]. Defaults to 'train'.
    """
    def decorator(func):
        # login logger and write running time 
        logged = logging.getLogger(log_name)

        @wraps(func)
        def wrapper(*args,**kwargs):
            # logged.info(desc.upper() + " :start running process {} ".format(func.__name__))
            s_time = time.time()
            ret = func(*args,**kwargs)
            e_time = time.time()
            logged.info("{} took {} seconds".format(desc, e_time - s_time))
            # logged.info("<<<< {} process is done".format(func.__name__))
            return ret

        @attach_wrapper(wrapper)
        def set_desc(describe):
            nonlocal desc 
            desc = describe
        
        @attach_wrapper(wrapper)
        def set_logger(newname):
            nonlocal logged
            logged = logging.getLogger(newname)

        return wrapper
    return decorator  

# 将上面的修改成简单的print的形式
def timethis(desc = 'train'):
    """通过logger的形式记录每段/监控程序的运行时间

    Args:
        desc (str): [用来帮助识别运行状态是训练还是测试等等]. Defaults to 'train'.
        log_name (str): [再外部定义好的logger的名称]]. Defaults to 'train'.
    """
    def decorator(func):
        # login logger and write running time 
        @wraps(func)
        def wrapper(*args,**kwargs):
            print(desc.upper() + " :start running process {} ".format(func.__name__))
            s_time = time.time()
            ret = func(*args,**kwargs)
            e_time = time.time()
            print("{} took {} seconds".format(func.__name__, e_time - s_time))
            print(" {} process is done".format(func.__name__))
            return ret
        return wrapper
    return decorator


def record_nbatch(log_name='train', writer=None, phaseName='Train Analysis', verbose=False): 
    def decorator(func):
        logged = logging.getLogger(log_name)
        @wraps(func)
        def wrapper(*args,**kwargs):
            ret = func(*args,**kwargs)
            assert writer is not None, "you should set up the writer in decorator first"
            assert isinstance(ret,dict), "you should make the output of this module DICT to \
                                            upload it to tensorboard"
            
            if 'epoch' not in phaseName: index = ret['count'] + ret['epoch'] * ret['itera']
            else: 
                index = ret['epoch']
            if not verbose: return ret
            
            for k,v in ret.items():             
                if 'loss' in k: 
                    writer.add_scalars('Loss/{}'.format(phaseName), {k:v}, index)
                elif 'acc' in k: 
                    writer.add_scalars('Accuracy/{}'.format(phaseName),{k:v},index)
                else: 
                    writer.add_scalars('Weights/{}'.format(phaseName+k), {k:v}, index)
                
            if 'epoch' in phaseName:
                logged.info("{} : {}".format(phaseName.upper(),ret))
            else:
                logged.debug("{} : iteration {} ==> {}".format(phaseName.upper(),index,ret))
            return ret

        @attach_wrapper(wrapper)
        def set_writer(newwriter):
            nonlocal writer
            writer = newwriter
        
        @attach_wrapper(wrapper)
        def set_logger(newname):
            nonlocal logged
            logged = logging.getLogger(newname)
        
        @attach_wrapper(wrapper)
        def set_phase(newphase):
            nonlocal phaseName
            phaseName = newphase
        
        @attach_wrapper(wrapper)
        def set_verbose(status):
            nonlocal verbose
            verbose = status

        return wrapper
    return decorator

def visual_1s(writer = None, phaseName = 'first_'):
    # mode， text， scalars， graph/image 
    def decorator(func):

        @wraps(func)
        def wrapper(*args,**kwargs):
            # logged.info("="*25 + ">visualize data in tensorboard")
            ret = func(*args,**kwargs)
            # write ret into tensorboard 
            assert writer is not None, "you should set up the writer in decorator first"
            # 检查类型是figure
            for i in range(len(ret)):
                writer.add_figure(phaseName, ret[i], i)
            # writer.add_figure(phaseName + 'visualize here', ret)
            return ret 

        @attach_wrapper(wrapper)
        def set_writer(newwriter):
            nonlocal writer
            writer = newwriter
        
        @attach_wrapper(wrapper)
        def set_phase(newphase):
            nonlocal phaseName
            phaseName = newphase

        return wrapper
    return decorator

def text_in_tensorboard(writer=None, Title='Record'):
    def decorator(func):

        @wraps(func)
        def wrapper(*args,**kwargs):
            # logged.info("="*25 + ">visualize data in tensorboard")
            ret = func(*args,**kwargs)
            # write ret into tensorboard 
            assert writer is not None, "you should set up the writer in decorator first"
            # 检查类型是figure
            assert isinstance(ret,str), "the return of this function should be a string"
            writer.add_text(Title, ret)
            # writer.add_figure(phaseName + 'visualize here', ret)
            return ret 

        @attach_wrapper(wrapper)
        def set_writer(newwriter):
            nonlocal writer
            writer = newwriter
        
        @attach_wrapper(wrapper)
        def set_title(newphase):
            nonlocal Title
            Title = newphase

        return wrapper
    return decorator

# 将装饰器修改成类的方法的话，虽然可以定义实例，但是我们是针对同一个Recorder需要调用不同的指令，所以实际上不是特别的贴切
# 但是如果使用这种方式的话应该要在函数之间传递装饰器的两个具体实例，val，train，test
# ============================TEST============================================================
@timethis()
def TestWrapper(n):
    for i in range(n):
        print(i)
        time.sleep(1)
    print("end testing ")
    return n

if __name__ == "__main__":
    print(__doc__)
    TestWrapper(5)


