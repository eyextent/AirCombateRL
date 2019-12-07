import numpy as np
import yaml
from easydict import EasyDict as edict
import os
import argparse
from common.utlis import judge_type
base_dir = os.path.dirname(os.path.dirname(__file__))
#print(base_dir)
dict = edict()


def cfg_from_file(subfolder,filename):
    if filename=='None':
        return

    with open(os.path.join(base_dir,"argument",subfolder,"{}.yaml".format(filename)),'r',encoding='utf-8') as f:
        yaml_cfg = edict(yaml.load(f,Loader=yaml.FullLoader))

    return yaml_cfg

def merge(param):
    for key,value in param.items():
        if param[key]=='None':
            continue
        else:
            dict.update(cfg_from_file(key,value))
   # print(dict)
    return dict

def args_wrapper_parser(args):
    '''
    将字典形式的参数解析到 argparse 中，方便命令行传参
    '''
    parser = argparse.ArgumentParser()
    for key, value in args.items():
        tp = judge_type(value)
        if tp == 'list':
            eval("parser.add_argument('--%s', default='%s')"%(key, value))
        else:
            eval("parser.add_argument('--%s', type=%s, default='%s')"%(key, tp, value))
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


# def cfg_from_file(filename,subfolder):
#     if subfolder==None:
#         with open(os.path.join(base_dir, "config","{}.yaml".format(filename)), 'r', encoding='utf-8') as f:
#             yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
#     else:
#         with open(os.path.join(base_dir,"config",subfolder,"{}.yaml".format(filename)),'r',encoding='utf-8') as f:
#             yaml_cfg = edict(yaml.load(f,Loader=yaml.FullLoader))
#
#     return yaml_cfg


#合并操作，以‘/’为分隔符进行分割文件夹和文件名

# def merge(param):
#     for key in param:
#         if key.find('/')==-1:
#             dict[key]=cfg_from_file(key,None)
#         else:
#             print(key.split('/')[0])
#             print(key.split('/')[1])
#             dict[key.split('/')[1]]=cfg_from_file(key.split('/')[1],key.split('/')[0])
#     return dict





