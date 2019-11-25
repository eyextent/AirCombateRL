#!usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import sys
sys.path.append('..')
import utlis.alloc as alloc
from argument.dqnArgs import args


def run_AirCombat_selfPlay(env, train_agent_list, use_agent_list, train_agent_name):  
    '''
    Params：
        env:                class object
        train_agent_list:   class object list
        use_agent_list:     class object list
        train_agent_name:   str

    主要逻辑：
        将红、蓝智能体分为训练智能体、使用智能体进行训练；
        使用 utlis.selfPlayUtlis模块 进行 红&蓝 与 训练&使用 之间的转换,
        完成训练和测试功能，并可以进行可视化。
    '''
    # todo
    raise NotImplementedError
    