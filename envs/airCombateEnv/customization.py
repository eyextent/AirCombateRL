import numpy as np

from argument.argManage import args

np.random.seed(args.env_random_seed)


# ===========================================
#                posture setting
#        including position and angle
# ===========================================
def init_posture(init_scen, red, blue, random_r, random_b):
    """
    param:
        init_scen:      场景类型
        red:            红方
        blue:           蓝方
        random_r:       红方是否随机；1：随机；0：固定
        random_b:       蓝方是否随机；1：随机；0：固定
    return:
        红方、蓝方
    主要逻辑：
        最外围逻辑判断根据场景类型进行区分
        --内层逻辑根据红蓝方是否随机区分
    """
    if init_scen == 0:  # 随机
        if random_r == 1:
            red.ac_pos = np.append(np.random.uniform(-500, 500),
                                   np.random.uniform(-250, 250))
            red.ac_heading = np.random.uniform(0, 360)
        elif random_r == 0:
            red.ac_pos = np.array([100.0, 0.0])  # 二维坐标
            red.ac_heading = 0  # 朝向角,向东
        else:
            raise Exception("random_r error")

        if random_b == 1:
            blue.ac_pos = np.append(np.random.uniform(-500, 500),
                                    np.random.uniform(-250, 250))
            blue.ac_heading = np.random.uniform(0, 360)
        elif random_b == 0:
            blue.ac_pos = np.array([-100.0, 0.0])  # 二维坐标
            blue.ac_heading = 180  # 朝向角,向东
        else:
            raise Exception("random_b error")
    elif init_scen == 1:  # 进攻
        if random_r == 1:
            red.ac_pos = np.append(np.random.uniform(100, 500),
                                   np.random.uniform(-50, 50))
            red.ac_heading = np.random.uniform(150, 210)
        elif random_r == 0:
            red.ac_pos = np.array([100.0, 0.0])  # 二维坐标
            red.ac_heading = 0  # 朝向角,向东
        else:
            raise Exception("random_r error")

        if random_b == 1:
            blue.ac_pos = np.append(np.random.uniform(-100, -500),
                                    np.random.uniform(-50, 50))
            # 330~360 0~30
            blue.ac_heading = np.random.uniform(0, 30)
        elif random_b == 0:
            blue.ac_pos = np.array([-100.0, 0.0])  # 二维坐标
            blue.ac_heading = 180  # 朝向角,向东
        else:
            raise Exception("random_b error")
    elif init_scen == 2:  # 防守
        if random_r == 1:
            red.ac_pos = np.append(np.random.uniform(100, 500),
                                   np.random.uniform(-50, 50))
            # 330~360 0~30
            red.ac_heading = np.random.uniform(0, 30)
        elif random_r == 0:
            red.ac_pos = np.array([100.0, 0.0])  # 二维坐标
            red.ac_heading = 180
        else:
            raise Exception("random_r error")

        if random_b == 1:
            blue.ac_pos = np.append(np.random.uniform(-100, -500),
                                    np.random.uniform(-50, 50))
            blue.ac_heading = np.random.uniform(150, 210)
        elif random_b == 0:
            blue.ac_pos = np.array([-100.0, 0.0])  # 二维坐标
            blue.ac_heading = 0
        else:
            raise Exception("random_b error")
    elif init_scen == 3:  # 同向(面对面)
        if random_r == 1:
            red.ac_pos = np.append(np.random.uniform(100, 500),
                                   np.random.uniform(-50, 50))
            red.ac_heading = np.random.uniform(150, 210)
        elif random_r == 0:
            red.ac_pos = np.array([100.0, 0.0])  # 二维坐标
            red.ac_heading = 180
        else:
            raise Exception("random_r error")

        if random_b == 1:
            blue.ac_pos = np.append(np.random.uniform(-100, -500),
                                    np.random.uniform(-50, 50))
            # 330~360, 0~30
            blue.ac_heading = np.random.uniform(0, 30)
        elif random_b == 0:
            blue.ac_pos = np.array([-100.0, 0.0])  # 二维坐标
            blue.ac_heading = 0
        else:
            raise Exception("random_b error")
    elif init_scen == 4:  # 中立(平行)
        if random_r == 1:
            red.ac_pos = np.append(np.random.uniform(-50, 50),
                                   np.random.uniform(-250, 250))
            red.ac_heading = 0
        elif random_r == 0:
            red.ac_pos = np.array([0.0, 100.0])
            red.ac_heading = 0
        else:
            raise Exception("random_r error")

        if random_b == 1:
            blue.ac_pos = np.append(np.random.uniform(-50, 50),
                                    np.random.uniform(-250, 250))
            blue.ac_heading = 0
        elif random_b == 0:
            blue.ac_pos = np.array([0.0, -100.0])
            blue.ac_heading = 0
        else:
            raise Exception("random_b error")
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
