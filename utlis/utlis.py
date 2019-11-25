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
#                   math
# ===========================================
def distance(agent1:list, agent2:list):
    dis = 0
    for pos1, pos2 in zip(agent1, agent2):
        dis = dis + (pos1 - pos2)*(pos1 - pos2)
    dis = math.sqrt(dis) 
    return dis