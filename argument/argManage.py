from common.config import merge
import os
param = {'env': 'airCombateEnv', 'algs': 'dqn', 'memory': 'memory'}  # memory不用的时候value为None即可
args_origin = merge(param)
def args_wrapper(args):
    # 主要是对重复训练的保存路径进行封装
    # None
    args_origin.save_path = args_origin.source_path + '/' + args_origin.experiment_name + '/'
    # args_origin.save_path = args_origin.source_path + '/' 'your name' + '/' + args_origin.experiment_name + '/'
    if not os.path.exists(args_origin.save_path):
    	os.makedirs(args_origin.save_path)
    return args

args = args_wrapper(args_origin)