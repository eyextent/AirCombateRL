import sys
sys.path.append("..")
from common.config import merge
from common.utlis import judge_type
import os
import argparse

param = {'env': 'airCombateEnv', 'algs': 'dqn', 'memory': 'memory'}  # memory不用的时候value为None即可
args_origin = merge(param)


def args_wrapper_parser(args):
    '''
    将字典形式的参数解析到 argparse 中，方便命令行传参
    '''
    parser = argparse.ArgumentParser()
    for key, value in args.items():
        eval("parser.add_argument('--%s', type=%s, default='%s')"%(key, judge_type(value), value))
    args = parser.parse_args()
    return args

def args_wrapper_path(args):
    '''
    主要是对重复训练的保存路径进行封装
    '''
    args.save_path = args.source_path + '/' + args.experiment_name + '/'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args

args_after_parse = args_wrapper_parser(args_origin)
args = args_wrapper_path(args_after_parse)
# print(args)