#!usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import time
import sys
sys.path.append('..')
from argument.dqnArgs import args
from envs.units import REGISTRY as registry_unints

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

np.set_printoptions(suppress=True)

class Env(object):
    def __init__(self):
        '''
        基类构造函数：
            还至少需要设置 state_space, action_space, state_dim, action_dim
            包含 单位实例列表 red_unit_list, blue_unit_list
        '''
        # 地图设置
        self.AREA = args.map_area  # 地图范围
        # 可视化显示参数
        self.SCALE = args.map_scale  # 比例尺
        self.SCOPE = self.AREA * self.SCALE  # Tkinter显示范围

        # 训练参数
        self.t = args.map_t  # 时间间隔（步长，秒)
        self.td = self.t / args.map_t_n  # 每个步长计算次数
        self.Sum_Oil = 100  # 油量，即每个episode的最大step数量

        ## 飞机列表
        self.red_unit_list = []
        self.blue_unit_list = []

    def _seed(self,seed):
        '''
        设置随机种子
        '''
        np.random.seed(seed)

    def reset(self):
        '''
        环境初始化
        输出：
            当前状态（所有飞机的当前状态）
        '''
        raise NotImplementedError

    def step(self, *action):
        '''
        输入：
            动作（环境内所有飞机的动作）
        输出：
            <下一状态，奖励值，done>，当前状态可以省略
        
        待扩展：
            局部可观察obs，额外状态信息（全局状态）state
        '''
        raise NotImplementedError

    def _get_reward(self,*args):
        '''
        根据当前飞机信息，给出奖励函数。
        最好是输入设置为当前环境能够给出的所有信息，返回的是奖励值。
        这样当使用时可以不用管中间的逻辑实现，只需看输入、输出；
        当想创建新的奖励函数模型时，只需要根据环境所能提供的信息，来实现此函数内逻辑的实现。
        '''
        raise NotImplementedError

    def render(self):
        '''
        环境的可视化显示
        '''
        raise NotImplementedError

    def close(self):
        '''
        可视化关闭
        '''
        raise NotImplementedError

    # Partially Observable Env for Multi-Agent
    # 部分可观察的多智能体环境使用
    def get_state(self):
        raise NotImplementedError

    def get_state_shape(self):
        raise NotImplementedError

    def get_agent_obs(self, agent_id):
        raise NotImplementedError

    def get_agent_obs_shape(self):
        raise NotImplementedError

    def _get_avail_actions(self):
        raise NotImplementedError

    def _get_agent_avail_actions(self, agent_id):        
        raise NotImplementedError   


class AirCombatEnvMultiUnit(Env):
    def __init__(self):
        super(AirCombatEnv, self).__init__()

        # 初始化双方飞机
        id_number = 0
        for name in args.red_unit_type_list:        # args.red_unit_type_list为飞机类型名字列表
            red_unit = registry_unints[name](id_number)  
            self.red_unit_list.append(red_unit)
            id_number = id_number + 1

        id_number = 0
        for name in args.blue_unit_type_list:        
            blue_unit = registry_unints[name](id_number)  
            self.blue_unit_list.append(blue_unit)
            id_number = id_number + 1

        self.n_red_unit = len(args.red_unit_type_list)
        self.n_blue_unit = len(args.blue_unit_type_list)

        # 强化学习动作接口
        # 【方案一】 直接使用联结动作空间，即 [a1, a2, ..., an, a1, a2, ..., an, ...]
        # 注意：如果红蓝双方飞机数量不一致，则分别定义
        self.single_action_space = ['l', 's', 'r']  # 向左滚转、维持滚转、向右滚转
        self.action_space = self.single_action_space * self.n_red_unit
        # # 【方案二】 使用decentralized policy，即动作空间只包含单独一个agent的
        # self.action_space = ['l', 's', 'r']

        self.action_dim = self.n_actions = len(self.action_space)
        self.state_dim = 5                   # 需要定义

        # todo：reward判断指标

        
    def reset_selfPlay(self):
        self.done = False
        self.success = 0
        self.acts = [[], []]  # todo-levin: 只保存一个agent的acts，还是两个acgent的acts？

        # 1°初始化红蓝红方飞机
            # todo: 
            # 【方案一】：
            # 按照实际阵型构建初始化阵型函数，初始化中心飞机的位置，确定每个飞机的位置
            # 【方案二】：
            # 随机初始化一个区域，在区域内随机初始各飞机的位置

            # 遍历列表初始化飞机的状态：滚转角、油量等
            
        # 2°得到初始状态
        for unit in self.red_unit_list:
            # 分别得到红、蓝双方的联结状态空间
            pass
        for unit in self.blue_unit_list:
            pass

        return s_b, s_r

    # levin - [done]： add both actions
    def step_selfPlay(self, action_blue_list, action_red_list):
        '''
        Parms:
            action_b: list or tuple
            action_r: list or tuple
        '''
        # 1° 双方飞机移动
        self._unit_move(self.blue_unit_list, action_blue_list)
        self._unit_move(self.red_unit_list, action_red_list)

       # 2° todo：得到联结状态空间
       # 3° todo：得到奖励值和done

    def _unit_move(self, unit_list, action_list):
        # todo：判断unit_list和action_list的维度匹配
        for unit, action in zip(unit_list, action_list):
            unit.move(action)
            
        