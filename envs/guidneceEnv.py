import numpy as np
import time
import sys
from envs.tools import init_pos
sys.path.append('..')
from argument.dqnArgs import args
from envs.unit import REGISTRY as registry_unit
from common.utlis import distance
from envs.airCombateEnv import Env
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

np.set_printoptions(suppress=True)

class GuidenceEnvOverload(Env):
    def __init__(self):
        super(GuidenceEnvOverload, self).__init__()
        # 飞机参数
        self.aircraft = registry_unit["overload"]()  # 创建飞机实例
        # 渐进点参数
        self.ap_pos = np.array([0, 0, 0])
        self.ap_heading = 0
        # state参数
        self.last_action = 1
        self.oil = args.Sum_Oil     # 总步数
        # reward判断指标
        self.dis_error = 200        # 距离误差
        self.angle_error = 5        # 角度误差
        self.altitude_error = 110   # 高度误差
        # 强化学习动作接口
        self.n_actions = len(self.aircraft.action_space)    # 动作空间
        self.action_dim = self.n_actions                    # 动作空间长度
        self.state_dim = len(self._get_state(self.aircraft, self.ap_pos, self.ap_heading, self.last_action))    # 状态空间长度
        # reward条件
        self.success = 0

    def reset(self):
        # 初始化参数
        self.success = 0
        self.acts = []
        self.last_action = 1
        self.oil = args.Sum_Oil
        # 初始化舰载机和最终进近点坐标、朝向
        self.aircraft, self.ap_pos, self.ap_heading = init_pos(self.aircraft, self.ap_pos, self.ap_heading)
        # 返回状态
        state = self._get_state(self.aircraft, self.ap_pos, self.ap_heading, self.last_action)
        return state

    def _get_reward(self, aircraft, ap_pos, ap_heading):
        """
        param:
            aircraft:               舰载机
            ap_pos:                 最终进近点坐标
            ap_heading:             最终进近点朝向
        return:
            回报值、该次训练结束标识
        主要逻辑：
            根据reward判定条件判定reward和done值并返回
        """
        # 计算舰载机坐标和最终进近点坐标水平距离
        dis = distance(aircraft.ac_pos, ap_pos)
        # reward
        if (abs(aircraft.ac_heading - ap_heading) <= self.angle_error) and (dis <= self.dis_error) \
                and (abs(aircraft.ac_pos[2] - ap_pos[2]) <= self.altitude_error):
            done = True
            self.success = 1
            reward = 20
        elif (self.oil <= 0):
            done = True
            reward = -10
        elif (aircraft.ac_pos[0] > args.map_area or 0 - aircraft.ac_pos[0] > args.map_area or
              aircraft.ac_pos[1] > args.map_area or 0 - aircraft.ac_pos[1] > args.map_area) or \
                (aircraft.ac_pos[2] > 8000) or (aircraft.ac_pos[2] < 0):
            done = True
            reward = -20
        else:
            reward = 0
            done = False
        return reward, done

    def step(self, action):
        self.acts.append(action)
        # 转化为[0~6]动作
        convert_action = self._transfer_action(action)
        # 每次move的t为0.1，共执行5次
        for i in range(args.map_t_n):
            self.aircraft.move(convert_action)
            reward, done = self._get_reward(self.aircraft, self.ap_pos, self.ap_heading)
            if done is True:
                break
        self.oil = self.oil - 1
        s = self._get_state(self.aircraft, self.ap_pos, self.ap_heading, self.last_action)
        self.last_action = action
        return s, reward, done

    def _get_state(self, aircraft, ap_pos, ap_heading, last_action):
        """
        param:
            aircraft:               舰载机
            ap_pos:                 最终进近点坐标
            ap_heading:             最终进近点朝向
            last_action:            上一次机动动作
        return:
            状态
        主要逻辑：
            根据舰载机和最终进近点坐标、朝向和舰载机油量，上一次机动动作计算状态值
        """
        state = np.concatenate(((aircraft.ac_pos - ap_pos) / args.map_area,
                                [(aircraft.ac_heading - ap_heading) / 180,
                                self.oil / args.Sum_Oil,
                                last_action / (self.n_actions - 1)]))
        return state

    def _transfer_action(self, action):
        """
        param:
            action:               对应场景类型的机动动作
        return:
            转化后[0~6]机动动作
        主要逻辑：
            根据场景类型将对应场景类型的机动动作转化为[0~6]机动动作
        """
        if args.envs_type == "2D_xy":
            # 0 1 2 3 4 -> 0 1 2 5 6
            if action == 3 or action == 4:
                action = action + 2
            return action
        elif args.envs_type == "2D_xz":
            # 0 1 2 3 4 -> 0 3 4 5 6
            if action != 0:
                action = action + 2
            return action
        elif args.envs_type == "3D":
            return action

    def creat_ALG(self):
        self.Tk = tk.Tk()
        self.Tk.title('1V1')
        self.Tk.canvas = tk.Canvas(self.Tk, bg='white',
                                   height=args.map_area * args.map_scale * 2,
                                   width=args.map_area * args.map_scale * 2)
        self.x_show = self.xyz2abc([0,0,0])
        self.x = self.Tk.canvas.create_oval(
            self.x_show[0] - 1, self.x_show[1] - 1,
            self.x_show[0] + 1, self.x_show[1] + 1,
            fill='black')
        self.Tk.canvas.pack()

    def render(self):
        # 刷新红方飞机
        self.r_show = self.xyz2abc(self.aircraft.ac_pos)
        self.r = self.Tk.canvas.create_oval(
            self.r_show[0] - 1, self.r_show[1] - 1,
            self.r_show[0] + 1, self.r_show[1] + 1,
            fill='red')

        self.Tk.update()
        time.sleep(0.05)
        # if self.done:
        #     time.sleep(0.1)
        #     self.Tk.destroy()

    def xyz2abc(self, pos):
        pos_show = np.array([0, 0])
        pos_show[0] = pos[0] * args.map_scale + args.map_area * args.map_scale
        pos_show[1] = args.map_area * args.map_scale - pos[1] * args.map_scale
        return pos_show

# 环境测试程序
if __name__ == '__main__':
    args.Sum_Oil = 100
    args.map_scale = 0.01
    args.map_area = 75000
    env = GuidenceEnvOverload()
    s = env.reset()
    env.creat_ALG()
    env.render()
    A = [1, 1, 2, 1, 2, 1, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 2, 4, 2, 4, 2, 4, 4, 2,
     0, 2, 2, 4, 4, 2, 0, 3, 3, 2, 2, 2, 2, 2, 2, 3, 2, 3 , 3, 2, 2, 4, 2, 2, 2, 2, 4, 4, 4, 1, 4, 4, 1, 4, 4, 4, 4, 4,
     4, 4, 4, 4, 2, 2, 2, 0, 4, 2, 2, 2, 2, 2, 2, 2, 1, 0, 2, 2, 1, 1, 1, 1, 1]
    for a in A:
        s_b, reward_r, done = env.step(a)
        print(s_b, reward_r, done)
        env.render()