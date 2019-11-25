import numpy as np
import math
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
np.set_printoptions(suppress=True)
np.random.seed(2)

G = 9.81
ROLL_RATE = 40                                  #滚转角变化率


class AirCombatEnv_new():
    def __init__(self):
        self.AREA = 3000                        # 地图范围

        # 可视化显示参数
        self.SCALE = 0.1  # 比例尺
        self.SCOPE = self.AREA * self.SCALE     #Tkinter显示范围

        # 飞机参数
        self.ac_speed = 200                     # 飞机速度，m/s
        self.ac_bank_angle_b_max = 80           #φ_blue_max = 80
        self.ac_bank_angle_r_max = 80           #φ_red_max = 60

        #训练参数
        self.t = 0.5                           #时间间隔（步长，秒)
        self.td = self.t / 5                    #每个步长计算次数
        self.Sum_Oil = 1000                      #油量，即每个episode的最大step数量

        #reward判断指标
        self.AA_range = 60                      #视界角范围
        self.ATA_range = 30                     #天线拂擦角范围
        self.Dis_max = 500               #距离最大值
        self.Dis_min = 100               #距离最小值
        self.adv_count = 0              #持续建立优势的次数

        #初始想定模式：0随机，1进攻，2防御，3同向，4中立
        self.init_scen = 0

        # 强化学习动作接口
        self.action_space = ['l','s','r']       #向左滚转、维持滚转、向右滚转
        self.n_actions = len(self.action_space)
        self.n_features = 3
        self.action_dim = self.n_actions
        self.state_dim = 6

        # self.flag_change_random_init = 1
        # self.random_init_pos_r = 1

    def reset_selfPlay(self):
        self.reward_b = 0
        self.reward_r = 0
        self.done = False
        self.success = 0
        self.acts = [[],[]]
        self.advs = []
        self.ATA_b = self.AA_b = 100
        self.ATA_r = self.AA_r = 100
        self.adv_count = 0

        #初始化红方飞机
        # self.ac_pos_r = np.array([0.0,0.0])         #二维坐标
        # self.ac_heading_r = np.array([0])                       #朝向角 向东
        self.ac_pos_b = np.array([0.0,0.0])         #二维坐标
        self.ac_heading_b = np.array([0])                       #朝向角 向东
        self.ac_bank_angle_r = 0                    #滚转角一开始应该都为0

        #初始化蓝方飞机（角度向东为0，逆时针旋转）
        if self.init_scen == 0:         #随机
            # self.ac_pos_b = np.append(np.random.uniform(-500,500,size=1),np.random.uniform(-250,250,size=1))
            # self.ac_heading_b = np.random.uniform(0,360,size=1)
            self.ac_pos_r = np.append(np.random.uniform(-500,500,size=1),np.random.uniform(-250,250,size=1))
            self.ac_heading_r = np.random.uniform(0,360,size=1)
        elif self.init_scen == 1:       #进攻
            self.ac_pos_b = np.append(np.random.uniform(-100,-500,size=1),np.random.uniform(-50,50,size=1))
            self.ac_heading_b = np.random.uniform(0,60,size=1)
        elif self.init_scen == 2:       #防守
            self.ac_pos_b = np.append(np.random.uniform(100,500,size=1),np.random.uniform(-50,50,size=1))
            self.ac_heading_b = np.random.uniform(0,30,size=1)
        elif self.init_scen == 3:       #同向
            self.ac_pos_b = np.append(np.random.uniform(100, 500, size=1), np.random.uniform(-50, 50, size=1))
            self.ac_heading_b = np.random.uniform(150, 180, size=1)
        else:                           #中立
            self.ac_pos_b = np.append(np.random.uniform(-50,50,size=1),np.random.uniform(-250,250,size=1))
            self.ac_heading_b = np.array([0])
        self.ac_bank_angle_b = 0  # 滚转角一开始应该都为0

        # # todo:双方飞机的初始位置和朝向角都应该包含随机和固定两种模式
        # if self.flag_change_random_init:
        #     self.ac_pos_r, self.ac_pos_b = self.ac_pos_b, self.ac_pos_r
        #     self.ac_heading_r, self.ac_heading_b = self.ac_heading_b, self.ac_heading_r


        self.oil = np.array([int(self.Sum_Oil)])

        dis = math.sqrt((self.ac_pos_r[0] - self.ac_pos_b[0]) * (self.ac_pos_r[0] - self.ac_pos_b[0])
                        + (self.ac_pos_r[1] - self.ac_pos_b[1]) * (self.ac_pos_r[1] - self.ac_pos_b[1]))

        # 计算ATA和AA
        heading_r = self.ac_heading_r
        heading_b = self.ac_heading_b

        self.ATA_b, self.AA_b = self.getAngle(self.ac_pos_r, self.ac_pos_b, heading_r, heading_b)
        self.ATA_r, self.AA_r = self.getAngle(self.ac_pos_b, self.ac_pos_r, heading_b, heading_r)

        #计算优势
        if (dis < self.Dis_max) and (dis > self.Dis_min) \
                and (abs(self.AA_b) < self.AA_range) and (abs(self.ATA_b) < self.ATA_range):
            if self.adv_count >= 0:     #如果之前蓝方已经是优势态势，优势累加
                self.adv_count += 1
            else:                       #如果之前是红方优势态势，优势交替
                self.adv_count = 1
        elif(dis < self.Dis_max) and (dis > self.Dis_min) \
                and (abs(self.AA_r) < self.AA_range) and (abs(self.ATA_r) < self.ATA_range):
            if self.adv_count <= 0:
                self.adv_count -= 1
            else:
                self.adv_count = -1
        else:
            self.adv_count = 0
        self.advs.append(self.adv_count)
        #reward shaping
        RA_b = 1 - ((1 - math.fabs(self.ATA_b) / 180) + (1 - math.fabs(self.AA_b) / 180))
        RA_r = 1 - ((1 - math.fabs(self.ATA_r) / 180) + (1 - math.fabs(self.AA_r) / 180))
        RD = math.exp(-(math.fabs(dis - ((self.Dis_max + self.Dis_min) / 2)) / 180 * 0.1))
        Rbl_b = RA_b * RD
        Rbl_r = RA_r * RD

        self.fai_b = -0.01 * Rbl_b
        self.fai_r = -0.01 * Rbl_r

        #返回红蓝飞机状态
        s_b = np.concatenate(((self.ac_pos_b-self.ac_pos_r)/self.AREA,
                            [self.ac_heading_b/180, self.ac_heading_r/180,
                             self.ac_bank_angle_b/80,self.adv_count/10]))
        s_r = np.concatenate(((self.ac_pos_r-self.ac_pos_b)/self.AREA,
                            [self.ac_heading_r/180, self.ac_heading_b/180,
                             self.ac_bank_angle_r/80, 0-self.adv_count/10]))
        return s_b, s_r

    def step_selfPlay(self, action_b, action_r):
        #存储红蓝飞机动作
        self.acts[0].append(action_b)
        self.acts[1].append(action_r)

        for i in range(5):
            #蓝方飞机运动计算
            self.ac_bank_angle_b = self.ac_bank_angle_b + (action_b - 1) * ROLL_RATE * self.td
            self.ac_bank_angle_b = max(self.ac_bank_angle_b,-self.ac_bank_angle_b_max)
            self.ac_bank_angle_b = min(self.ac_bank_angle_b, self.ac_bank_angle_b_max)
            self.turn_rate = (G / self.ac_speed) * math.tan(self.ac_bank_angle_b * math.pi / 180) * 180 / math.pi
            self.ac_heading_b = self.ac_heading_b - self.turn_rate * self.td
            if self.ac_heading_b > 360:
                self.ac_heading_b -= 360
            elif self.ac_heading_b < 0:
                self.ac_heading_b += 360
            self.ac_pos_b[0] = self.ac_pos_b[0] + self.ac_speed * self.td * math.cos(self.ac_heading_b * math.pi / 180)
            self.ac_pos_b[1] = self.ac_pos_b[1] + self.ac_speed * self.td * math.sin(self.ac_heading_b * math.pi / 180)

            #红方飞机运动计算
            self.ac_bank_angle_r = self.ac_bank_angle_r + (action_r - 1) * ROLL_RATE * self.td
            self.ac_bank_angle_r = max(self.ac_bank_angle_r, -self.ac_bank_angle_r_max)
            self.ac_bank_angle_r = min(self.ac_bank_angle_r, self.ac_bank_angle_r_max)
            self.turn_rate = (G / self.ac_speed) * math.tan(self.ac_bank_angle_r * math.pi / 180) * 180 / math.pi
            self.ac_heading_r = self.ac_heading_r - self.turn_rate * self.td
            if self.ac_heading_r > 360:
                self.ac_heading_r -= 360
            elif self.ac_heading_r < 0:
                self.ac_heading_r += 360
            self.ac_pos_r[0] = self.ac_pos_r[0] + self.ac_speed * self.td * math.cos(self.ac_heading_r * math.pi / 180)
            self.ac_pos_r[1] = self.ac_pos_r[1] + self.ac_speed * self.td * math.sin(self.ac_heading_r * math.pi / 180)

        self.oil -= 1

        #返回红蓝飞机状态
        s_b = np.concatenate(((self.ac_pos_b-self.ac_pos_r)/self.AREA,
                            [self.ac_heading_b/180, self.ac_heading_r/180,
                             self.ac_bank_angle_b/80,self.adv_count/10]))
        s_r = np.concatenate(((self.ac_pos_r-self.ac_pos_b)/self.AREA,
                            [self.ac_heading_r/180, self.ac_heading_b/180,
                             self.ac_bank_angle_r/80, 0-self.adv_count/10]))

        dis = math.sqrt((self.ac_pos_r[0] - self.ac_pos_b[0]) * (self.ac_pos_r[0] - self.ac_pos_b[0])
                        + (self.ac_pos_r[1] - self.ac_pos_b[1]) * (self.ac_pos_r[1] - self.ac_pos_b[1]))

        #计算ATA和AA
        heading_r = self.ac_heading_r
        heading_b = self.ac_heading_b

        self.ATA_b, self.AA_b = self.getAngle(self.ac_pos_r, self.ac_pos_b, heading_r, heading_b)
        self.ATA_r, self.AA_r = self.getAngle(self.ac_pos_b, self.ac_pos_r, heading_b, heading_r)

        #计算优势
        if (dis < self.Dis_max) and (dis > self.Dis_min) \
                and (abs(self.AA_b) < self.AA_range) and (abs(self.ATA_b) < self.ATA_range):
            if self.adv_count >= 0:     #如果之前蓝方已经是优势态势，优势累加
                self.adv_count += 1
            else:                       #如果之前是红方优势态势，优势交替
                self.adv_count = 1
        elif(dis < self.Dis_max) and (dis > self.Dis_min) \
                and (abs(self.AA_r) < self.AA_range) and (abs(self.ATA_r) < self.ATA_range):
            if self.adv_count <= 0:
                self.adv_count -= 1
            else:
                self.adv_count = -1
        else:
            self.adv_count = 0
        self.advs.append(self.adv_count)

        #reward shaping
        RA_b = 1 - ((1 - math.fabs(self.ATA_b) / 180) + (1 - math.fabs(self.AA_b) / 180))
        RA_r = 1 - ((1 - math.fabs(self.ATA_r) / 180) + (1 - math.fabs(self.AA_r) / 180))
        RD = math.exp(-(math.fabs(dis - ((self.Dis_max + self.Dis_min) / 2)) / 180 * 0.1))
        Rbl_b = RA_b * RD
        Rbl_r = RA_r * RD
        self.old_fai_b = self.fai_b
        self.old_fai_r = self.fai_r
        self.fai_b = -0.01 * Rbl_b
        self.fai_r = -0.01 * Rbl_r

        # 计算reward和终止条件
        if self.adv_count >= 9:
            self.done = True
            self.success = 1
            self.reward_b = 2.0
            self.reward_r = -2.0
        elif self.adv_count <= -9:
            self.done = True
            self.success = -1
            self.reward_b = -2.0
            self.reward_r = 2.0
        elif (self.oil <= 0):
            self.done = True
            self.success = 0
            self.reward_b = -1.0
            self.reward_r = -1.0
        elif (self.ac_pos_b[0] > self.AREA) or ((0 - self.ac_pos_b[0]) > self.AREA) or \
                (self.ac_pos_b[1] > self.AREA) or ((0 - self.ac_pos_b[1]) > self.AREA):
            self.done = True
            self.success = 0
            self.reward_b = -1.0
            self.reward_r = (self.fai_r - self.old_fai_r) - 0.001
        elif (self.ac_pos_r[0] > self.AREA) or ((0 - self.ac_pos_r[0]) > self.AREA) or \
                (self.ac_pos_r[1] > self.AREA) or ((0 - self.ac_pos_r[1]) > self.AREA):
            self.done = True
            self.success = 0
            self.reward_b = (self.fai_b - self.old_fai_b) - 0.001
            self.reward_r = -1.0
        else:
            self.done = False
            self.reward_b = (self.fai_b - self.old_fai_b) - 0.001
            self.reward_r = (self.fai_r - self.old_fai_r) - 0.001

        return s_b, s_r, self.reward_b, self.reward_r, self.done

    def creat_ALG(self):
        self.Tk = tk.Tk()
        self.Tk.title('1V1')
        self.Tk.canvas = tk.Canvas(self.Tk, bg='white',
                                height=self.SCOPE*2,
                                width=self.SCOPE*2)
        self.Tk.canvas.pack()

    def render(self):
        #刷新红方飞机
        self.r_show = self.xyz2abc(self.ac_pos_r)
        self.r = self.Tk.canvas.create_oval(
            self.r_show[0] - 1, self.r_show[1] - 1,
            self.r_show[0] + 1, self.r_show[1] + 1,
            fill='red')

        #刷新蓝方飞机
        self.b_show = self.xyz2abc(self.ac_pos_b)
        self.b =  self.Tk.canvas.create_oval(
            self.b_show[0] - 1, self.b_show[1] - 1,
            self.b_show[0] + 1, self.b_show[1] + 1,
            fill='blue')

        self.Tk.update()
        time.sleep(0.05)
        if self.done:
            time.sleep(0.1)
            self.Tk.destroy()

    def getAngle(self,pos_r, pos_b, heading_r, heading_b):              #计算AA和ATA
        theta_br = 180 * math.atan2((pos_r[1] - pos_b[1]), (pos_r[0] - pos_b[0])) / math.pi
        theta_rb = 180 * math.atan2((pos_b[1] - pos_r[1]), (pos_b[0] - pos_r[0])) / math.pi
        if theta_br < 0:
            theta_br = 360 + theta_br
        if theta_rb < 0:
            theta_rb = 360 + theta_rb
        ATA = heading_b - theta_br
        AA = 180 + heading_r - theta_rb
        if ATA > 180:
            ATA = 360 - ATA
        elif ATA < -180:
            ATA = 360 + ATA
        if AA > 180:
            AA = 360 - AA
        elif AA < -180:
            AA = 360 + AA
        return ATA, AA

    def xyz2abc(self,pos):
        pos_show = np.array([0,0])
        pos_show[0] = pos[0] * self.SCALE + self.SCOPE
        pos_show[1] = self.SCOPE - pos[1] * self.SCALE
        return pos_show

# #环境测试程序
# if __name__ == '__main__':
#     env = AirCombatEnv()
#     s = env.reset_selfPlay()
#     env.creat_ALG()
#     env.render()
#     while True:
#         if env.done:
#             print('-------------------------------------------------')
#             s = env.reset_selfPlay()
#             env.creat_ALG()
#         #a = np.random.randint(0,3)
#         a_b = 2
#         a_r = 2
#         env.step_selfPlay(a_b,a_r)
#         env.render()






class AirCombatEnv(Env):
    def __init__(self):
        super(AirCombatEnv, self).__init__()

        # 飞机参数
        self.red = registry_unints["default"](200,80)  #红方飞机
        self.red_unit_list.append(self.red)

        self.blue = registry_unints["default"](200,80)  #蓝方飞机
        self.blue_unit_list.append(self.blue)

        # 强化学习动作接口
        # todo：应该每个飞机的动作空间放入unit类中？
        self.action_space = ['l', 's', 'r']  # 向左滚转、维持滚转、向右滚转
        self.n_actions = len(self.action_space)

        self.action_dim = self.n_actions
        self.state_dim = 5

        # reward判断指标
        self.AA_range = 60  # 视界角范围
        self.ATA_range = 30  # 天线拂擦角范围

        self.Dis_ERROR_max = 500  # 距离最大值
        self.Dis_ERROR_min = 100  # 距离最小值

    def reset_selfPlay(self):
        self.acts = [[], []]  # todo-levin: 只保存一个agent的acts，还是两个acgent的acts？

        # 初始化红方飞机
        self.red.ac_pos = np.array([0.0, 0.0])  # 二维坐标
        self.red.ac_heading = 180  # 朝向角 向南
        self.red.ac_bank_angle = 0  # 滚转角一开始应该都为0
        self.red.oil = np.array([int(self.Sum_Oil)])
        # 初始化蓝方飞机
        self.blue.ac_pos = np.append(np.random.uniform(-1000, 1000, size=1), np.random.uniform(-1500, -1000, size=1))
        self.blue.ac_heading = np.random.uniform(-180, 180, size=1)
        self.blue.ac_bank_angle = 0  # 滚转角一开始应该都为0
        self.blue.oil = np.array([int(self.Sum_Oil)])

        dis = math.sqrt((self.red.ac_pos[0] - self.blue.ac_pos[0]) * (self.red.ac_pos[0] - self.blue.ac_pos[0])
                        + (self.red.ac_pos[1] - self.blue.ac_pos[1]) * (self.red.ac_pos[1] - self.blue.ac_pos[1]))
        self.ATA_b, self.AA_b, self.ATA_r, self.AA_r = self._get_AA_ATA(self.red.ac_pos,self.red.ac_heading,
                                                                        self.blue.ac_pos, self.blue.ac_heading)

        ho_b = ((1 - math.fabs(self.ATA_b) / 180) + (1 - math.fabs(self.AA_b) / 180)) / 2
        ho_r = ((1 - math.fabs(self.ATA_r) / 180) + (1 - math.fabs(self.AA_r) / 180)) / 2
        hd = -(math.fabs(dis - self.Dis_ERROR_max) + math.fabs(dis - self.Dis_ERROR_min)) \
             / math.fabs(dis + self.Dis_ERROR_max + self.Dis_ERROR_min)
        self.fai_b = ho_b + hd
        self.fai_r = ho_r + hd

        s_b = np.concatenate(((self.blue.ac_pos - self.red.ac_pos) / self.AREA,
                              [self.blue.ac_heading / 180, self.red.ac_heading / 180, self.blue.ac_bank_angle / 80]))

        s_r = np.concatenate(((self.red.ac_pos - self.blue.ac_pos) / self.AREA,
                              [self.red.ac_heading / 180, self.blue.ac_heading / 180, self.red.ac_bank_angle / 80]))
        return s_b, s_r

    # levin - [done]： add both actions
    def step_selfPlay(self, action_b, action_r):
        self.acts[0].append(action_b)
        self.acts[1].append(action_r)

        self.red.move(action_r)
        self.blue.move(action_b)

        # todo-levin: s逻辑的设置是否需要改变!!!  ATA, AA 是代表什么!! reward的逻辑是否需要改变!

        s_b = np.concatenate(((self.blue.ac_pos - self.red.ac_pos) / self.AREA,
                              [self.blue.ac_heading / 180, self.red.ac_heading / 180, self.blue.ac_bank_angle / 80]))
        s_r = np.concatenate(((self.red.ac_pos - self.blue.ac_pos) / self.AREA,
                              [self.red.ac_heading / 180, self.blue.ac_heading / 180, self.red.ac_bank_angle / 80]))
        reward_b, reward_r, done = self._get_reward(self.red.ac_pos,self.red.ac_heading,
                                                    self.blue.ac_pos, self.blue.ac_heading)
        return s_b, s_r, reward_b, reward_r, done

    def _get_reward(self,ac_pos_r, ac_heading_r, ac_pos_b, ac_heading_b):
        dis = math.sqrt((ac_pos_r[0] - ac_pos_b[0]) * (ac_pos_r[0] - ac_pos_b[0])
                        + (ac_pos_r[1] - ac_pos_b[1]) * (ac_pos_r[1] - ac_pos_b[1]))

        self.ATA_b, self.AA_b, self.ATA_r, self.AA_r = self._get_AA_ATA(ac_pos_r,ac_heading_r,
                                                                        ac_pos_b, ac_heading_b)

        ho_b = ((1 - math.fabs(self.ATA_b) / 180) + (1 - math.fabs(self.AA_b) / 180)) / 2
        ho_r = ((1 - math.fabs(self.ATA_r) / 180) + (1 - math.fabs(self.AA_r) / 180)) / 2
        hd = -(math.fabs(dis - self.Dis_ERROR_max) + math.fabs(dis - self.Dis_ERROR_min)) \
             / math.fabs(dis + self.Dis_ERROR_max + self.Dis_ERROR_min)
        self.old_fai_b = self.fai_b
        self.old_fai_r = self.fai_r
        self.fai_b = ho_b + hd
        self.fai_r = ho_r + hd
        # 计算reward和终止条件
        if (dis < self.Dis_ERROR_max) and (dis > self.Dis_ERROR_min) \
                and (abs(self.AA_b) < self.AA_range) and (abs(self.ATA_b) < self.ATA_range):
            self.done = True
            self.success = 1
            self.reward_b = 1.0
            self.reward_r = 0.0
        elif (dis < self.Dis_ERROR_max) and (dis > self.Dis_ERROR_min) \
                and (abs(self.AA_r) < self.AA_range) and (abs(self.ATA_r) < self.ATA_range):
            self.done = True
            self.success = -1
            self.reward_b = 0.0
            self.reward_r = 1.0
        elif (self.red.oil <= 0 or self.blue.oil <= 0):
            self.done = True
            self.success = 0
            self.reward_b = 0.0
            self.reward_r = 0.0
        elif (ac_pos_b[0] > self.AREA) or ((0 - ac_pos_b[0]) > self.AREA) or \
                (ac_pos_b[1] > self.AREA) or ((0 - ac_pos_b[1]) > self.AREA) or \
                (ac_pos_r[0] > self.AREA) or ((0 - ac_pos_r[0]) > self.AREA) or \
                (ac_pos_r[1] > self.AREA) or ((0 - ac_pos_r[1]) > self.AREA):
            self.done = True
            self.success = 0
            self.reward_b = 0.0
            self.reward_r = 0.0
        else:
            self.done = False
            self.reward_b = self.fai_b - self.old_fai_b
            self.reward_r = self.fai_r - self.old_fai_r
        return self.reward_b, self.reward_r, self.done

    def _get_AA_ATA(self,ac_pos_r, ac_heading_r, ac_pos_b,ac_heading_b):
        if ac_heading_r < 0:
            ac_heading_r += 360
        if ac_heading_b < 0:
            ac_heading_b += 360
        # 计算ATA和AA
        if ac_heading_b >= 0 and ac_heading_b < 90:
            heading_b = 90 - ac_heading_b
        else:
            heading_b = 450 - ac_heading_b
        if ac_heading_r >= 0 and ac_heading_r < 90:
            heading_r = 90 - ac_heading_r
        else:
            heading_r = 450 - ac_heading_r

        ATA_b, AA_b = self.getAngle(ac_pos_r, ac_pos_b, heading_r, heading_b)
        ATA_r, AA_r = self.getAngle(ac_pos_b, ac_pos_r, heading_b, heading_r)
        return ATA_b, AA_b, ATA_r, AA_r

    def creat_ALG(self):
        self.Tk = tk.Tk()
        self.Tk.title('1V1')
        self.Tk.canvas = tk.Canvas(self.Tk, bg='white',
                                   height=self.SCOPE * 2,
                                   width=self.SCOPE * 2)
        self.Tk.canvas.pack()

    def render(self):
        # 刷新红方飞机
        self.r_show = self.xyz2abc(self.red.ac_pos)
        self.r = self.Tk.canvas.create_oval(
            self.r_show[0] - 1, self.r_show[1] - 1,
            self.r_show[0] + 1, self.r_show[1] + 1,
            fill='red')

        # 刷新蓝方飞机
        self.b_show = self.xyz2abc(self.blue.ac_pos)
        self.b = self.Tk.canvas.create_oval(
            self.b_show[0] - 1, self.b_show[1] - 1,
            self.b_show[0] + 1, self.b_show[1] + 1,
            fill='blue')

        self.Tk.update()
        time.sleep(0.05)
        if self.done:
            time.sleep(0.1)
            self.Tk.destroy()

    def getAngle(self, pos_r, pos_b, heading_r, heading_b):  # 计算AA和ATA
        theta_br = 180 * math.atan2((pos_r[1] - pos_b[1]), (pos_r[0] - pos_b[0])) / math.pi
        theta_rb = 180 * math.atan2((pos_b[1] - pos_r[1]), (pos_b[0] - pos_r[0])) / math.pi
        # print(pos_b,pos_r,theta_br,theta_rb)
        if theta_br < 0:
            theta_br = 360 + theta_br
        if theta_rb < 0:
            theta_rb = 360 + theta_rb
        ATA = heading_b - theta_br
        AA = 180 + heading_r - theta_rb
        if ATA > 180:
            ATA = 360 - ATA
        elif ATA < -180:
            ATA = 360 + ATA
        if AA > 180:
            AA = 360 - AA
        elif AA < -180:
            AA = 360 + AA
        return ATA, AA

    def xyz2abc(self, pos):
        pos_show = np.array([0, 0])
        pos_show[0] = pos[0] * self.SCALE + self.SCOPE
        pos_show[1] = self.SCOPE - pos[1] * self.SCALE
        return pos_show
