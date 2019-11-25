#!usr/bin/env python3
# -*- coding: utf-8 -*-

'''
定义各种飞机及其他单位信息:
定义每种类型飞机的 锁定区域半径、状态格式、（锁定时间）、（可视范围）、动作空间及运动模型等
''' 
import math
import numpy as np
import sys
sys.path.append('..')
from argument.dqnArgs import args

class Aircraft(object):
    def __init__(self, id_number=None):
        '''
        需要定义飞机的属性，飞机的编号
        考虑：每个单位的动作空间，状态空间
        '''
        raise NotImplementedError 

    def move(self, action):
        '''
        使用动力学模型：_aricraft_dynamic
            输入：动作
            输出：位置，朝向角等
        '''
        raise NotImplementedError

    def get_oil(self):
        '''
        剩余油量
        '''
        raise NotImplementedError

    def attack_range(self):
        '''
        攻击范围
        '''
        raise NotImplementedError

    def locking_time(self):
        '''
        锁定时间
        '''
        raise NotImplementedError     


# 子飞机类型设计
class AircraftDefault(Aircraft):

    def __init__(self, id=None, ac_speed=200, ac_bank_angle_max=80):
        self.ac_pos = np.array([0.0, 0.0])                                  # 二维坐标
        self.ac_heading = 180                                               # 朝向角
        self.ac_bank_angle = 0                                              # 滚转角
        self.ac_bank_angle_max = ac_bank_angle_max                          # φ_max
        self.ac_speed = ac_speed                                            # 飞机速度，m/s
        self.oil = 100                                                      # 油量，即每个episode的最大step数量
        self.td = 0.1

    def move(self, action):
        for i in range(5):
            #飞机运动计算
            self.ac_bank_angle = self.ac_bank_angle + (action - 1) * args.roll_rate * self.td
            self.ac_bank_angle = max(self.ac_bank_angle,-self.ac_bank_angle_max)
            self.ac_bank_angle = min(self.ac_bank_angle, self.ac_bank_angle_max)
            self.turn_rate = (args.G / self.ac_speed) * math.tan(self.ac_bank_angle * math.pi / 180) * 180 / math.pi
            self.ac_heading = self.ac_heading + self.turn_rate * self.td
            if self.ac_heading > 360:
                self.ac_heading -= 360
            elif self.ac_heading < 0:
                self.ac_heading += 360
            #print("ac_heading_b:", self.ac_heading_b)
            self.ac_pos[0] = self.ac_pos[0] + self.ac_speed * self.td * math.sin(self.ac_heading * math.pi / 180)
            self.ac_pos[1] = self.ac_pos[1] + self.ac_speed * self.td * math.cos(self.ac_heading * math.pi / 180)
        self.oil -= 1

    def get_oil(self):
        return self.oil


REGISTRY = {}
REGISTRY["default"] = AircraftDefault
#REGISTRY["name_new"] = NewClass