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
        self.ac_pos = np.array([0.0, 0.0])  # 二维坐标
        self.ac_heading = 0  # 朝向角,向南
        self.ac_bank_angle = 0  # 滚转角一开始应该都为0
        self.oil = args.Sum_Oil
        self.ac_bank_angle_max = ac_bank_angle_max  # φ_max
        self.ac_speed = ac_speed  # 飞机速度，m/s
        self.td = args.map_t / args.map_t_n

    def move(self, action):
        for i in range(args.map_t_n):
            # 飞机运动计算
            self.ac_bank_angle = self.ac_bank_angle + (action - 1) * args.roll_rate * self.td
            self.ac_bank_angle = max(self.ac_bank_angle, -self.ac_bank_angle_max)
            self.ac_bank_angle = min(self.ac_bank_angle, self.ac_bank_angle_max)
            self.turn_rate = (args.G / self.ac_speed) * math.tan(self.ac_bank_angle * math.pi / 180) * 180 / math.pi
            self.ac_heading = self.ac_heading - self.turn_rate * self.td
            if self.ac_heading > 360:
                self.ac_heading -= 360
            elif self.ac_heading < 0:
                self.ac_heading += 360
            self.ac_pos[0] = self.ac_pos[0] + self.ac_speed * self.td * math.cos(self.ac_heading * math.pi / 180)
            self.ac_pos[1] = self.ac_pos[1] + self.ac_speed * self.td * math.sin(self.ac_heading * math.pi / 180)
        self.oil -= 1

    def get_oil(self):
        return self.oil


class AircraftOverload(Aircraft):
    def __init__(self, ac_speed=150):
        # 根据场景类型设置动作空间
        if args.envs_type == "2D_xy":
            self.action_space = ['s', 'l', 'r', 'a', 'd']
        elif args.envs_type == "2D_xz":
            self.action_space = ['s','u','n','a','d']
        elif args.envs_type == "3D":
            self.action_space = ['s','l','r','u','n','a','d']
        else:
            raise Exception("envs_type error")
        self.ac_pos = np.array([0.0, 0.0, 5000.0])  # 三维坐标
        self.ac_speed = ac_speed                    # 飞机速度，m/s
        self.ac_speed_min = 100                     # 最大飞行速度
        self.ac_speed_max = 300                     # 最小飞行速度
        self.ac_heading = 0                         # 朝向角/偏角
        self.ac_pitch = 0                           # 俯仰角/倾角
        self.ac_pitch_max = 60
        self.ac_roll = 0                            # 滚转角
        self.rate_roll = 40                         # 滚转角变化率
        self.roll_max = 80                          # 滚转角最大值
        self.oil = args.Sum_Oil                     # 飞机油量
        self.t = args.map_t / args.map_t_n          # 时间片
        self.nx = 0                                 # 切向过载
        self.ny = 1                                 # 过载
        self.nz = 0                                 # 过载
        self.nf = 5                                 # 法向过载
        # 测试参数
        self.list = []
        self.x = []
        self.y = []
        self.z = []

    def move(self, action):
        # 设置nx、ny、nz值
        self._overload(action)
        # 计算t时刻后加速度、倾角变化率、偏角变化率
        a, rate_pitch, rate_heading = self._get_rate(self.nx, self.ny, self.nz, self.ac_pitch, self.ac_speed)
        # 根据运动学公式计算t时刻后速度、倾角、偏角、滚转角
        # 速度
        self.ac_speed = self.ac_speed + a * self.t
        ##### 检查速度
        self.ac_speed = max(self.ac_speed, self.ac_speed_min)
        self.ac_speed = min(self.ac_speed, self.ac_speed_max)
        #####
        # 倾角
        last_pitch = self.ac_pitch
        self.ac_pitch = self.ac_pitch + rate_pitch * 180 * self.t / math.pi
        ##### 检查倾角 /////////记录变化之前和变化之后相乘为负即归零，判断是不是爬升俯冲
        if last_pitch * self.ac_pitch < 0 and (action != 3 or action != 4):
            self.ac_pitch = 0
        if action == 3:     # 飞机在执行爬升动作时，检查俯仰角是否超出范围
            self.ac_pitch = min(self.ac_pitch, self.ac_pitch_max)
        elif action == 4:   # 飞机在执行俯冲动作时，检查俯仰角是否超出范围
            self.ac_pitch = max(self.ac_pitch, -self.ac_pitch_max)
        #####
        # 偏角
        self.ac_heading = self.ac_heading + rate_heading * 180 * self.t / math.pi
        ##### 检查偏角
        if self.ac_heading > 180:
            self.ac_heading = self.ac_heading - 360
        elif self.ac_heading < -180:
            self.ac_heading = self.ac_heading + 360
        #####
        # 滚转角
        last_roll = self.ac_roll
        self.ac_roll = self.ac_roll + self.rate_roll * self.t
        ##### 检查滚转角
        self.ac_roll = max(self.ac_roll, -self.roll_max)
        self.ac_roll = min(self.ac_roll, self.roll_max)
        if last_roll * self.ac_roll < 0 and (action != 1 or action != 2):
            self.ac_roll = 0
        #####
        # 根据运动学公式计算t时刻后x、y、z坐标
        self.ac_pos[0] = self.ac_pos[0] + self.ac_speed * math.cos(self.ac_pitch * math.pi / 180) * math.cos(
            self.ac_heading * math.pi / 180) * self.t
        self.ac_pos[1] = self.ac_pos[1] + -1 * self.ac_speed * math.cos(self.ac_pitch * math.pi / 180) * math.sin(
            self.ac_heading * math.pi / 180) * self.t
        self.ac_pos[2] = self.ac_pos[2] + self.ac_speed * math.sin(self.ac_pitch * math.pi / 180) * self.t
        # 绘图用参数
        self.x.append(self.ac_pos[0])
        self.y.append(self.ac_pos[1])
        self.z.append(self.ac_pos[2])

    def _get_rate(self, nx, ny, nz, pitch, v):
        """
        param:
            nx:                 切向过载
            ny:                 沿y轴方向过载
            nz:                 沿z轴方向过载
            pitch：             倾角
            v:                  速度
        return:
            加速度、倾角变化率、偏角变化率
        主要逻辑：
            根据运动学公式计算t时刻后加速度、倾角变化率、偏角变化率
        """
        # 飞机加速度
        a = args.G * (nx - math.sin(pitch * math.pi / 180))
        # 倾角的变化率
        rate_pitch = (ny - math.cos(pitch * math.pi / 180)) * args.G / v
        # 偏角的变化率,考虑倾角为90情况
        if pitch == 90 or pitch == -90:  # θ = 90 or -90
            rate_heading = 0
        else:
            rate_heading = 1 * nz * args.G / (v * math.cos(pitch * math.pi / 180))
        return a, rate_pitch, rate_heading

    def _overload(self, action):
        """
        param:
            action:               机动动作
        主要逻辑：
            根据飞机滚转角、俯仰角、法向过载和机动动作设置nx、ny、nz的值
        """
        ##### 检查滚转角
        if self.ac_roll < 0:        # 刚执行完左转动作，需要将滚转角归零
            self.rate_roll = 40
        elif self.ac_roll > 0:
            self.rate_roll = -40    # 刚执行完右转动作，需要将滚转角归零
        else:
            self.rate_roll = 0      # 滚转角为0，无需改变
        ##### 检查倾角
        if self.ac_pitch < 0:       # 刚执行完俯冲动作，需要将倾角归零
            self.ny = self.nf * math.cos(self.ac_roll * math.pi / 180)
        elif self.ac_pitch > 0:     # 刚执行完爬升动作，需要将倾角归零
            self.ny = - self.nf * math.cos(self.ac_roll * math.pi / 180)
        else:
            self.ny = math.cos(self.ac_pitch * math.pi / 180)  # 倾角为0，rate_θ = 0
        #####
        ##### 检查nz
        if self.ac_pitch == 0:      # 倾角为0，nz为0
            self.nz = 0
        else:                       # 倾角不为0，nz会随滚转角值变化
            self.nz = self.nf * math.sin(self.ac_roll * math.pi / 180)
        #####
        if action == 0:             # 稳定飞行
            self.nx = math.sin(self.ac_pitch * math.pi / 180)  # 切向过载
        elif action == 1:           # 最大过载左转
            self.nx = math.sin(self.ac_pitch * math.pi / 180)
            if self.ac_pitch == 0:
                self.nz = -math.sqrt(self.nf * self.nf - self.ny * self.ny)
            self.rate_roll = -40
        elif action == 2:           # 最大过载右转
            self.nx = math.sin(self.ac_pitch * math.pi / 180)
            if self.ac_pitch == 0:
                self.nz = math.sqrt(self.nf * self.nf - self.ny * self.ny)
            self.rate_roll = 40
        elif action == 3:           # 最大过载爬升
            self.nx = math.sin(self.ac_pitch * math.pi / 180)
            self.ny = self.nf * math.cos(self.ac_roll * math.pi / 180)
        elif (action == 4):         # 最大过载俯冲
            self.nx = math.sin(self.ac_pitch * math.pi / 180)
            self.ny = -self.nf * math.cos(self.ac_roll * math.pi / 180)
        elif (action == 5):         # 最大加速
            self.nx = 2
        elif (action == 6):         # 最大减速
            self.nx = -2

    def show(self,name):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.plot3D(self.x, self.y, self.z, 'gray')  # 绘制空间曲线
        # ax1.set_xlim([-x, x])
        # ax1.set_ylim([-x, x])
        # ax1.set_zlim([5000-x, 5000+x])
        plt.savefig(name)
        plt.show()


REGISTRY = {}
REGISTRY["default"] = AircraftDefault
REGISTRY["overload"] = AircraftOverload
# REGISTRY["name_new"] = NewClass

# import  csv
# if __name__ == '__main__':
#     fileHeader = ["x", "y", "z", "heading", "pitch", "roll", "spd"]
#     for i in range(7):
#         for j in range(7):
#             if i == j:
#                 continue
#             A = [i,j]
#             name = "[" + str(A[0]) + "," + str(A[1]) + "]"
#             csvFile = open(name + ".csv", "w", newline='')
#             writer = csv.writer(csvFile)
#             writer.writerow(fileHeader)
#             envs = AircraftOverload()
#             line = [envs.ac_pos[0], envs.ac_pos[1], envs.ac_pos[2], envs.ac_heading, envs.ac_pitch, envs.ac_roll, envs.ac_speed]
#             writer.writerow(line)
#             for a in A:
#                 for k in range(5):
#                     envs.move(a)
#                     line = [envs.ac_pos[0], envs.ac_pos[1], envs.ac_pos[2], envs.ac_heading, envs.ac_pitch, envs.ac_roll,
#                             envs.ac_speed]
#                     writer.writerow(line)
#             csvFile.close()
#             envs.show(name)