#from runner.selfPlayNANU import output
from common.utlis import set_seed
from common.config import merge
from argument.argManage import args
import torch
# from config import yaml_cfg_all

if __name__ == "__main__":
    '''
    参数传递；
    参数打印；
    高阶逻辑；
    等...
    '''
    print(torch.__version__)
    # param = ["env1/envConfig","algConfig","env1/env1Config"]
    print(args.convs[1])
    print(args.learning_rate)
    print(args.memoryname)
    torch.
    # run()
