import numpy as np
from argument.dqnArgs import args

np.random.seed(args.env_random_seed)


def random_pos(init_scen, red, blue, random_r, random_b):
    """
    根据初始想定模式和是否随机对飞机位置进行初始化
    :param init_scen:初始想定模式
    :param red:
    :param blue:
    :param random_r:红方是否随机
    :param random_b:蓝方是否随机
    :return:red, blue
    """
    if random_r == 0 and random_b == 0:
        # 初始化A飞机
        red.ac_pos = np.array([250.0, 250.0])  # 二维坐标
        red.ac_heading = 0  # 朝向角,向东
        # 初始化B飞机
        blue.ac_pos = np.array([-250.0, -250.0])
        blue.ac_heading = 0
    elif random_r == 0 and random_b == 1:
        # 初始化A飞机
        red.ac_pos = np.array([0.0, 0.0])  # 二维坐标
        red.ac_heading = 0  # 朝向角,向东
        # 初始化B飞机
        random_aircraft_pos(init_scen, blue)
    elif random_r == 1 and random_b == 0:
        # 初始化A飞机
        blue.ac_pos = np.array([0.0, 0.0])  # 二维坐标
        blue.ac_heading = 0  # 朝向角,向东
        # 初始化B飞机
        random_aircraft_pos(init_scen, red)
    elif random_r == 1 and random_b == 1:
        if init_scen == 0:
            # 初始化A飞机
            red.ac_pos = np.append(np.random.uniform(-500, 500),
                                   np.random.uniform(-250, 250))
            red.ac_heading = np.random.uniform(0, 360)
            # 初始化B飞机
            blue.ac_pos = np.append(np.random.uniform(-500, 500),
                                    np.random.uniform(-250, 250))
            blue.ac_heading = np.random.uniform(0, 360)
        elif init_scen == 1:
            # 初始化A飞机
            red.ac_pos = np.append(np.random.uniform(100, 500),
                                   np.random.uniform(-50, 50))
            red.ac_heading = np.random.uniform(0, 60)
            # 初始化B飞机
            blue.ac_pos = np.append(np.random.uniform(-100, -500),
                                    np.random.uniform(-50, 50))
            blue.ac_heading = np.random.uniform(0, 60)
        elif init_scen == 2:
            # 初始化A飞机
            red.ac_pos = np.append(np.random.uniform(-100, -500),
                                   np.random.uniform(-50, 50))
            red.ac_heading = np.random.uniform(0, 30)
            # 初始化B飞机
            blue.ac_pos = np.append(np.random.uniform(100, 500),
                                    np.random.uniform(-50, 50))
            blue.ac_heading = np.random.uniform(0, 30)
        elif init_scen == 3:
            # 初始化A飞机
            red.ac_pos = np.append(np.random.uniform(100, 500),
                                   np.random.uniform(-50, 50))
            red.ac_heading = np.random.uniform(150, 180)
            # 初始化B飞机
            blue.ac_pos = np.append(np.random.uniform(-100, -500),
                                    np.random.uniform(-50, 50))
            blue.ac_heading = np.random.uniform(0, 30)
        elif init_scen == 4:
            # 初始化A飞机
            red.ac_pos = np.append(np.random.uniform(-50, 50),
                                   np.random.uniform(-250, 250))
            red.ac_heading = 0  # 朝向角,向东
            # 初始化B飞机
            blue.ac_pos = np.append(np.random.uniform(-50, 50),
                                    np.random.uniform(-250, 250))
            blue.ac_heading = 0
    else:
        raise Exception("random_r and random_b error")
    red.ac_bank_angle = 0  # 滚转角
    blue.ac_bank_angle = 0
    red.oil = args.Sum_Oil  # 油量
    blue.oil = args.Sum_Oil
    return red, blue


def random_aircraft_pos(init_scen, aircraft):
    """
    根据初始想定模式对飞机坐标进行初始化
    :param init_scen:初始想定模式
    :param aircraft:
    :return:aircraft
    """
    if init_scen == 0:
        # 初始化B飞机
        aircraft.ac_pos = np.append(np.random.uniform(-500, 500),
                                    np.random.uniform(-250, 250))
        aircraft.ac_heading = np.random.uniform(0, 360)
    elif init_scen == 1:
        # 初始化B飞机
        aircraft.ac_pos = np.append(np.random.uniform(-100, -500),
                                    np.random.uniform(-50, 50))
        aircraft.ac_heading = np.random.uniform(0, 60)
    elif init_scen == 2:
        # 初始化B飞机
        aircraft.ac_pos = np.append(np.random.uniform(100, 500),
                                    np.random.uniform(-50, 50))
        aircraft.ac_heading = np.random.uniform(0, 30)
    elif init_scen == 3:
        # 初始化B飞机
        aircraft.ac_pos = np.append(np.random.uniform(100, 500),
                                    np.random.uniform(-50, 50))
        aircraft.ac_heading = np.random.uniform(150, 180)
    elif init_scen == 4:
        # 初始化B飞机
        aircraft.ac_pos = np.append(np.random.uniform(-50, 50),
                                    np.random.uniform(-250, 250))
        aircraft.ac_heading = 0
    else:
        raise Exception("init_scen error")
    return aircraft


def init_pos(aircraft, ap_pos, envs_type):
    """
    根据场景类型初始化飞机和进近点坐标
    :param aircraft:飞机
    :param ap_pos:进近点
    :param envs_type:场景类型
    :return:aircraft, pos
    """
    area = args.map_area * 0.8
    if envs_type == "2D_xy":
        aircraft.ac_pos = np.array([np.random.randint(0, args.map_area*0.5),
                                   np.random.randint(-args.map_area*0.5,args.map_area*0.5)])

        aircraft.ac_pos = np.append(aircraft.ac_pos, 0)
        # aircraft.ac_pos = np.array([-10000.0, 0.0, 0.0])
        ap_pos = np.array([0.0, 0.0, 0.0])
    elif envs_type == "3D_xz":
        # aircraft.ac_pos = np.array(np.random.uniform(-area, area),
        #                             0,
        #                             np.random.uniform(0, 5000))
        ap_pos = np.array([0.0, 0.0, 0.0])
    elif envs_type == "3D":
        # aircraft.ac_pos = np.array(np.random.uniform(-area, area),
        #                             np.random.uniform(-area, area),
        #                             5000)
        ap_pos = np.array([0.0, 0.0, 0.0])
    else:
        raise Exception("envs_type error")
    return aircraft, ap_pos



# ===========================================
#                state setting
# ===========================================

def get_state(aircraft_a, aircraft_b, adv_count):
    """
    计算aircraft_b的状态
    :param aircraft_a:
    :param aircraft_b:
    :param adv_count:优势次数
    :return:aircraft_b的状态
    """
    state = np.concatenate(((aircraft_b.ac_pos - aircraft_a.ac_pos) / args.map_area,
                            [aircraft_b.ac_heading / 180, aircraft_a.ac_heading / 180,
                             aircraft_b.ac_bank_angle / 80, adv_count / 10]))
    return state

def get_state_direct_pos(aircraft_a, aircraft_b, adv_count):
    state = np.concatenate(((aircraft_b.ac_pos - aircraft_a.ac_pos) / args.map_area,
                            aircraft_b.ac_pos/args.map_area, aircraft_a.ac_pos/args.map_area,
                           [aircraft_b.ac_heading / 180, aircraft_a.ac_heading / 180,
                             aircraft_b.ac_bank_angle / 80, adv_count / 10]))
    return state

REGISTRY_STATE = {}
REGISTRY_STATE['orign_state'] = get_state
REGISTRY_STATE['state_direct_pos'] = get_state_direct_pos