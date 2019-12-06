#from runner.selfPlayNANU import output
#from common.utlis import set_seed
from common.config import merge
# from config import yaml_cfg_all
from easydict import EasyDict as edict

if __name__ == "__main__":
    '''
    参数传递；
    参数打印；
    高阶逻辑；
    等...
    '''
    arg = edict()
    # param = ["env1/envConfig","algConfig","env1/env1Config"]
    param = {'env': 'airCombateEnv', 'algs': 'dqn', 'memory': 'memory'}
    arg = merge(param)
    print(arg.name)
    print(arg.memoryname)
    # run()
