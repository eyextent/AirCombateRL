import numpy as np
from argument.dqnArgs import args

np.random.seed(args.env_random_seed)


def init_pos(aircraft, ap_pos, ap_heading):
    """
    param:
        aircraft:               舰载机
        ap_pos:                 最终进近点坐标
        ap_heading:             最终进近点朝向
    return:
        舰载机和最终进近点坐标、朝向
    主要逻辑：
        根据场景类型，初始化舰载机和最终进近点坐标、朝向
    """
    area = args.map_area * 0.8
    if args.envs_type == "2D_xy":
        aircraft.ac_pos = np.array([np.random.randint(-area, 0),
                                    np.random.randint(-area, area),
                                    0])
        ap_pos = np.array([0.0, 0.0, 0.0])
    elif args.envs_type == "2D_xz":
        aircraft.ac_pos = np.array([np.random.randint(-area, area),
                                   0,
                                   5000])
        ap_pos = np.array([0.0, 0.0, 0.0])
    elif args.envs_type == "3D":
        aircraft.ac_pos = np.array([np.random.randint(-area, area),
                                   np.random.randint(-area, area),
                                   5000])
        ap_pos = np.array([0.0, 0.0, 0.0])
    else:
        raise Exception("envs_type error")
    aircraft.ac_heading = 0
    ap_heading = 0
    return aircraft, ap_pos, ap_heading