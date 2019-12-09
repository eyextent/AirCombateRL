#!usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import torch
import torch.autograd as autograd


def set_seed(seed):
    import numpy as np
    import random
    import torch
    
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

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if is_cuda() else autograd.Variable(*args, **kwargs)


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
def distance(agent1:list, agent2:list):
    dis = 0
    for pos1, pos2 in zip(agent1, agent2):
        dis = dis + (pos1 - pos2)*(pos1 - pos2)
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
        raise Exception("Invalid type!", arg)

