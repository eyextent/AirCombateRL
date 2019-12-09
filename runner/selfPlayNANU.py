#!usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
import envs
from models.dqn import DQN2013 as DQN
#from argument.dqnArgs import args
import common.alloc as alloc
from trainer.episodeSelfPlayTrainer import run_AirCombat_selfPlay
from argument.argManage import args
# 使用的参数：
args.n_blue
args.n_red

# env拿到的单位列表：
env.blue_unit_list
env.red_unit_list
## 每个agent要添加一个序号
unit.number

# 定义
blue_agent_list = []
red_agent_list = []
for unit in env.blue_unit_list:
    pass

def creat_n_agent(unit_list, is_train, scope, sess):
    agent_list = []
    for unit in unit_list:
        new_agent = DQN(unit.state_dim, unit.action_dim, scope+str(unit.number), sess)
        agent.append(new_agent)
    return agent_list

def load_parms_n_agent(agent_list):
    for agent in agent_list:
        agent.load_parms()
        
def main():
    env = envs.make("airCobate")

    flag_is_train = args.flag_is_train    # flag_is_train = 1, 一个训练，一个使用； flag_is_train = 0, 两个都是在使用（根据train_agent状态决定输出谁的信息---flag_train_blue）
    flag_train_blue = args.flag_train_blue    # flag_train_blue = 1 时训练agent_blue； flag_train_blue = 0 时训练agent_red

    # todo: 创建多个agent，并传入 interactor的NANU 中
    raise NotImplementedError


