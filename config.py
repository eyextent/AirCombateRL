import numpy as np
import yaml
from easydict import EasyDict as edict
import os
base_dir = os.path.dirname(__file__)
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
    print(dict)
    return dict
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





