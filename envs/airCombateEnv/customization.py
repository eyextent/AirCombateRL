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
    if init_scen == 0:  # 随机
        if random_r == 0 and random_b == 1:  # red不随机,blue随机
            red.ac_pos = np.array([0.0, 0.0])  # 二维坐标
            red.ac_heading = 0  # 朝向角,向东
            blue.ac_pos = np.append(np.random.uniform(-500, 500),
                                    np.random.uniform(-250, 250))
            blue.ac_heading = np.random.uniform(0, 360)
        elif random_r == 1 and random_b == 0:  # red随机,blue不随机
            red.ac_pos = np.append(np.random.uniform(-500, 500),
                                   np.random.uniform(-250, 250))
            red.ac_heading = np.random.uniform(0, 360)
            blue.ac_pos = np.array([0.0, 0.0])  # 二维坐标
            blue.ac_heading = 0  # 朝向角,向东
        elif random_r == 1 and random_b == 1:  # red随机,blue随机
            red.ac_pos = np.append(np.random.uniform(-500, 500),
                                   np.random.uniform(-250, 250))
            red.ac_heading = np.random.uniform(0, 360)
            blue.ac_pos = np.append(np.random.uniform(-500, 500),
                                    np.random.uniform(-250, 250))
            blue.ac_heading = np.random.uniform(0, 360)
        else:
            raise Exception("random_r or random_b error")
    elif init_scen == 1:  # 进攻
        if random_r == 0 and random_b == 1:  # red不随机,blue随机
            red.ac_pos = np.array([0.0, 0.0])  # 二维坐标
            red.ac_heading = 0  # 朝向角,向东
            blue.ac_pos = np.append(np.random.uniform(-100, -500),
                                    np.random.uniform(-50, 50))
            blue.ac_heading = np.random.uniform(0, 30)
        elif random_r == 1 and random_b == 0:  # red随机,blue不随机
            red.ac_pos = np.append(np.random.uniform(-100, -500),
                                   np.random.uniform(-50, 50))
            red.ac_heading = np.random.uniform(0, 30)
            blue.ac_pos = np.array([0.0, 0.0])  # 二维坐标
            blue.ac_heading = 0  # 朝向角,向东
        elif random_r == 1 and random_b == 1:  # red随机,blue随机
            red.ac_pos = np.append(np.random.uniform(100, 500),
                                   np.random.uniform(-50, 50))
            red.ac_heading = np.random.uniform(0, 30)
            blue.ac_pos = np.append(np.random.uniform(-100, -500),
                                    np.random.uniform(-50, 50))
            blue.ac_heading = np.random.uniform(0, 30)
        else:
            raise Exception("random_r or random_b error")
    elif init_scen == 2:  # 防守
        if random_r == 0 and random_b == 1:  # red不随机,blue随机
            red.ac_pos = np.array([0.0, 0.0])  # 二维坐标
            red.ac_heading = 0  # 朝向角,向东
            blue.ac_pos = np.append(np.random.uniform(100, 500),
                                    np.random.uniform(-50, 50))
            blue.ac_heading = np.random.uniform(0, 30)
        elif random_r == 1 and random_b == 0:
            red.ac_pos = np.append(np.random.uniform(100, 500),
                                   np.random.uniform(-50, 50))
            red.ac_heading = np.random.uniform(0, 30)
            blue.ac_pos = np.array([0.0, 0.0])  # 二维坐标
            blue.ac_heading = 0  # 朝向角,向东
        elif random_r == 1 and random_b == 1:
            red.ac_pos = np.append(np.random.uniform(-100, -500),
                                   np.random.uniform(-50, 50))
            red.ac_heading = np.random.uniform(0, 30)
            blue.ac_pos = np.append(np.random.uniform(100, 500),
                                    np.random.uniform(-50, 50))
            blue.ac_heading = np.random.uniform(0, 30)
        else:
            raise Exception("random_r or random_b error")
    elif init_scen == 3:  # 同向(面对面)
        if random_r == 0 and random_b == 1:  # red不随机,blue随机
            red.ac_pos = np.array([0.0, 0.0])  # 二维坐标
            red.ac_heading = 0  # 朝向角,向东
            blue.ac_pos = np.append(np.random.uniform(100, 500),
                                    np.random.uniform(-50, 50))
            blue.ac_heading = np.random.uniform(150, 180)
        elif random_r == 1 and random_b == 0:
            red.ac_pos = np.append(np.random.uniform(100, 500),
                                   np.random.uniform(-50, 50))
            red.ac_heading = np.random.uniform(150, 180)
            blue.ac_pos = np.array([0.0, 0.0])  # 二维坐标
            blue.ac_heading = 0  # 朝向角,向东
        elif random_r == 1 and random_b == 1:
            red.ac_pos = np.append(np.random.uniform(100, 500),
                                   np.random.uniform(-50, 50))
            red.ac_heading = np.random.uniform(150, 180)
            blue.ac_pos = np.append(np.random.uniform(-100, -500),
                                    np.random.uniform(-50, 50))
            blue.ac_heading = np.random.uniform(0, 30)
        else:
            raise Exception("random_r or random_b error")
        pass
    elif init_scen == 4:  # 中立(平行)
        if random_r == 0 and random_b == 1:  # red不随机,blue随机
            red.ac_pos = np.array([0.0, 0.0])  # 二维坐标
            red.ac_heading = 0  # 朝向角,向东
            blue.ac_pos = np.append(np.random.uniform(-50, 50),
                                    np.random.uniform(-250, 250))
            blue.ac_heading = 0
        elif random_r == 1 and random_b == 0:
            red.ac_pos = np.append(np.random.uniform(-50, 50),
                                   np.random.uniform(-250, 250))
            red.ac_heading = 0  # 朝向角,向东
            blue.ac_pos = np.array([0.0, 0.0])  # 二维坐标
            blue.ac_heading = 0  # 朝向角,向东
        elif random_r == 1 and random_b == 1:
            red.ac_pos = np.append(np.random.uniform(-50, 50),
                                   np.random.uniform(-250, 250))
            red.ac_heading = 0  # 朝向角,向东
            blue.ac_pos = np.array([0.0, 0.0])  # 二维坐标
            blue.ac_heading = 0  # 朝向角,向东
        else:
            raise Exception("random_r or random_b error")
    else:
        raise Exception("init_scen error")
    return red, blue

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
                            aircraft_b.ac_pos / args.map_area, aircraft_a.ac_pos / args.map_area,
                            [aircraft_b.ac_heading / 180, aircraft_a.ac_heading / 180,
                             aircraft_b.ac_bank_angle / 80, adv_count / 10]))
    return state


REGISTRY_STATE = {}
REGISTRY_STATE['orign_state'] = get_state
REGISTRY_STATE['state_direct_pos'] = get_state_direct_pos
