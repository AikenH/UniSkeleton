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
            logged.info(desc.upper() + " :start running process {} >>>".format(func.__name__))
            s_time = time.time()
            ret = func(*args,**kwargs)
            e_time = time.time()
            logged.info(">>>{} took {} seconds".format(func.__name__, e_time - s_time))
            logged.info("<<<< {} process is done".format(func.__name__))
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
            print(desc.upper() + " :start running process {} >>>".format(func.__name__))
            s_time = time.time()
            ret = func(*args,**kwargs)
            e_time = time.time()
            print(">>>{} took {} seconds".format(func.__name__, e_time - s_time))
            print("<<<< {} process is done".format(func.__name__))
            return ret
        return wrapper
    return decorator

# [x]编写接受并记录函数输出的装饰器也就是训练过程中的每个循环的输出
# 和tensorboard结合使用，也就是接受函数的输出，然后直接进行一个tensorboard的记录，但是就是要将writer传入logger

def record_nbatch(log_name='train', writer = None, phaseName = 'Train Analysis ', consoleShow = True, mode = 'scalars'):
    # 这里writer可以使用直接传入的方式，实际上还有一种是用地址然后通过summary writer进行读取的方式
    # 后续如果有需求可以在上面进行简单的改变即可
    # mode， text， scalars， graph/image 
    def decorator(func):
        logged = logging.getLogger(log_name)
        # exceptKey = [ ] # can exclude some key to tensorboard
        @wraps(func)
        def wrapper(*args,**kwargs):
            # logged.info("="*25 + ">visualize data in tensorboard")
            ret = func(*args,**kwargs)
            # write ret into tensorboard 
            assert writer is not None, "you should set up the writer in decorator first"

            
            # 如果我们需要传入scalars的情况下，ret默认应该是dict
            assert isinstance(ret,dict), "you should make the output of this module DICT to \
                                            upload it to tensorboard"
            
            if 'epoch' not in phaseName:
                index = ret['count'] + ret['epoch'] * ret['itera']
            else:
                index = ret['epoch']
            
            for k,v in ret.items():
                # if k in exceptKey: continue
                
                if 'loss' in k:
                    writer.add_scalars('Loss/{}'.format(phaseName), {k:v}, index)
                # if k is something u want independen figure， make rule to change phaseName
                # such as loss and acc is separated
                # elif 'acc' in k or 'confi' in k:
                elif 'acc' in k:
                    writer.add_scalars('Accuracy/{}'.format(phaseName),{k:v},index)
                else:
                    writer.add_scalars('Weights/{}'.format(phaseName+k), {k:v}, index)

                # show console info here
                # logged.info("iteration {} ==> {} : {}".format(index, k, v))
            logged.debug(">>{} : iteration {} ==> {}".format(phaseName.upper(),index,ret))
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
        def set_show(status):
            nonlocal consoleShow
            consoleShow = status
        
        @attach_wrapper(wrapper)
        def set_mode(newmode):
            nonlocal mode
            mode = newmode

        return wrapper
    return decorator

# TODO: add second to add _img for feature map
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

# 将装饰器修改成类的方法的话，虽然可以定义实例，但是我们是针对同一个Recorder需要调用不同的指令，所以实际上不是特别的贴切
# 但是如果使用这种方式的话应该要在函数之间传递装饰器的两个具体实例，val，train，test
""" class Decorator_all:
    def __init__(self, logname='Train', writer=None, phaseName='Train Analysis',mode='scalar',consoleShow=True):
        self.logname = logname
        self.logged = logging.getLogger(logname)
        self.writer = writer
        self.phaseName = phaseName
        self.consoleShow = consoleShow

    def record_nbatch(self,func):
        # self.logged = logging.getLogger(logname)
        def wrapper(*args,**kwargs):
            # logged.info("="*25 + ">visualize data in tensorboard")
            ret = func(*args,**kwargs)
            # write ret into tensorboard 
            assert self.writer is not None, "you should set up the writer in decorator first"
            
            if self.mode == 'epoch output':
                self.phaseName = 'EPOCH ' + self.phaseName
            # 如果我们需要传入scalars的情况下，ret默认应该是dict
            assert isinstance(ret,dict), "you should make the output of this module DICT to \
                                            upload it to tensorboard"
            index = ret['count']
            for k,v in ret.items():
                if 'loss' in k:
                    self.writer.add_scalars(self.phaseName + 'loss', {k:v}, index)
                # if k is something u want independen figure， make rule to change self.phaseName
                # such as loss and acc is separated
                else:
                    self.writer.add_scalars(self.phaseName + k, {k:v}, index)
                # show console info here
                # logged.info("iteration {} ==> {} : {}".format(index, k, v))
            self.logged.info(">>{} : iteration {} ==> {}".format(self.phaseName.upper(),index,ret))

            # elif(self.mode == 'epoch output'):
            #     index = ret['count']
            #     for k,v in ret.items():
            #         # show console info here
            #         self.logged.info(">>{} : iteration {} ==> {} : {}".format(phaseName.upper(),index, k, v))
            # elif(self.mode == 'figure'):
            #     # maybe we can use this or not (maybe not necessary)
            #     writer.add_figure(phaseName,ret)
                
            # logged.info("finish this <"+ "="*25)
            return ret
        return wrapper

    def visual_1s(self,func):
        def wrapper(*args,**kwargs):
            ret = func(*args,**kwargs)
            # visualization for ones 
            self.logged.info(">>VISUALIZE PART ( for once) <<")
            assert self.writer is not None, "you should set up the writer in decorator first"
            # 验证数据类型-修改
            assert ret is not None, "you shoule pass type figure here"
            self.writer.add_figure(self.phaseName + 'Visualize here', ret)
            return ret 
        return wrapper
    
    
    # 为装饰器设置属性的，灵活变换装饰器的状态的部分
    def set_writer(self,newwriter):
        self.writer = newwriter
    
    def set_logger(self,newname):
        self.logged = logging.getLogger(newname)
    
    def set_phase(self,newphase):
        self.phaseName = newphase
    
    def set_show(self,status):
        self.consoleShow = status
    
    def set_mode(self,newmode):
        self.mode = newmode

# GLOBAL SETTING FOR DECORATOR 
train_dec = Decorator_all(phaseName='training  ')
val_dec = Decorator_all(phaseName='valiing  ')
test_dec = Decorator_all(phaseName='testing  ')
decorator_dict = {'train': train_dec, 'val': val_dec, 'test':test_dec} """


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


