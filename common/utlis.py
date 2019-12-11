#!usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import torch
import torch.autograd as autograd
import numpy as np


def set_seed(seed):
    import random

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ===========================================
#                   pytorch
# =========================================== 
def is_cuda():
    return torch.cuda.is_available()


Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if is_cuda() else autograd.Variable(*args,
                                                                                                                 **kwargs)


# ===========================================
#                 algorithm
# =========================================== 
def epsilon_scheduler(eps_start, eps_final, eps_decay):
    def function(frame_idx):
        return eps_final + (eps_start - eps_final) * math.exp(-1. * frame_idx / eps_decay)

    return function

# ===========================================
#                   math
# ===========================================
def distance(agent1: list, agent2: list):
    dis = 0
    for pos1, pos2 in zip(agent1, agent2):
        dis = dis + (pos1 - pos2) * (pos1 - pos2)
    dis = math.sqrt(dis)
    return dis


# ===========================================
#                   others
# ===========================================
def judge_type(arg):
    if isinstance(arg, str):
        return 'str'
    elif isinstance(arg, int):
        return 'int'
    elif isinstance(arg, float):
        return 'float'
    elif isinstance(arg, bool):
        return 'int'
    elif isinstance(arg, list):
        return 'list'
    else:
        raise Exception("Invalid type! %s"%str(type(arg)), arg)

def random_two_range(x1, y1, x2, y2):
    """
    param:
        x1:               范围1下限
        y1:               范围1上限
        x2:               范围2下限
        y2:               范围2上限
    return:
        在范围1和范围2中随机生成的整数
    主要逻辑：
        在范围1中随机生成一个整数one
        在范围2中随机生成一个整数two
        随机生成一个0~1的浮点数epsilon
        epsilon<0.5返回整数one
        epsilon>0.5返回整数two
    """
    one = np.random.uniform(x1, y1)
    two = np.random.uniform(x2, y2)
    epsilon = np.random.rand()
    if epsilon < 0.5:
        return one
    else:
        return two
