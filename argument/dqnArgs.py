<<<<<<< HEAD
#!usr/bin/env python3
# -*- coding: utf-8 -*-
'''
定义和DQN算法相关的超参数
'''

import argparse
import os
import pprint

parser = argparse.ArgumentParser()

# runner
parser.add_argument("--env_name", type=str, default="airCombate", help="游戏名字") 
parser.add_argument("--flag_is_train", type=int, default="1", help="flag_IsTrain = 1, 一个训练，一个使用； flag_IsTrain = 0, 两个都是在使用")
parser.add_argument("--flag_focus_blue", type=int, default="1", help="flag_focus_blue = 1 时训练agent_blue； flag_focus_blue = 0 时训练agent_red")


# interactor
parser.add_argument("--episode", type=int, default="2000000", help="训练的最大episode数")
parser.add_argument("--store", type=int, default="10000", help="初始化经验池时运行的episode数")
parser.add_argument("--test_episode", type=int, default="100", help="测试时运行的episode数")
parser.add_argument("--train_episode", type=int, default="100", help="每训练多少个episode后启动测试")

# models
parser.add_argument("--net_frame", type=str, default='mlp', help="[mlp, cnn2mlp, cnn2rnn2mlp]")
parser.add_argument("--hidden_units", default=[128, 128], help="每个隐藏层的神经元数量")
parser.add_argument("--convs", default=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], help="每个卷积层设置")
parser.add_argument("--batch_size", type=int, default="256")
parser.add_argument("--gamma", type=float, default="0.99")
parser.add_argument("--learning_rate", type=float, default="0.00005", help="学习率")
parser.add_argument("--initial_epsilon", type=float, default="1.0", help="初始探索概率")
parser.add_argument("--decay_rate", type=float, default="1.0", help="探索率下降幅度")
parser.add_argument("--replay_size", type=int, default="100000")


# envs
parser.add_argument("--unit_type", type=str, default='default',help="[default,]")
parser.add_argument("--red_unit_type_list", default=['default'], help="飞机类型包括：[default,]")  # MvN环境中使用
parser.add_argument("--blue_unit_type_list", default=['default'], help="飞机类型包括：[default,]") # MvN环境中使用

parser.add_argument("--envs_type", type=str, default='2D_xy',help="场景类型：2D_xy, 2D_xz, 3D")

parser.add_argument("--map_area", type=int, default='3000',help="设定地图范围")
parser.add_argument("--map_scale", type=float, default='0.1',help="地图比例尺")
parser.add_argument("--map_t", type=float, default='0.5',help="时间间隔（步长，秒)")
parser.add_argument("--map_t_n", type=int, default='5',help="每个步长计算次数")
parser.add_argument("--env_random_seed", type=int, default='2',help="环境随机种子")

parser.add_argument("--init_scen", type=int, default='0',help="初始想定模式包括：0随机，1进攻，2防御，3同向，4中立")
parser.add_argument("--random_r", type=int, default='0',help="红方是否在初始化过程中随机")
parser.add_argument("--random_b", type=int, default='1',help="蓝方是否在初始化过程中随机")

parser.add_argument("--G", type=float, default='9.81',help="重力加速度")
parser.add_argument("--roll_rate", type=int, default='40',help="滚转角变化率")

parser.add_argument("--random_init_pos_b", type=int, default='1',help="[0 or 1]")
parser.add_argument("--random_init_pos_r", type=int, default='1',help="[0 or 1]")

# unit
parser.add_argument("--Sum_Oil", type=int, default='1000',help="油量，即每个episode的最大step数量")

# utlis
parser.add_argument("--seed", type=int, default="125")

parser.add_argument("--source_path", type=str, default='../result',help="保存路径")                   
parser.add_argument("--experiment_name", type=str, default='blue_red_game', help="保存文件夹的名字")
parser.add_argument("--checkpoint_folder_name", type=str, default='_saved_networks/', help="参数保存文件夹的名字，加/结尾")
parser.add_argument("--file_name", type=str, default='_agent.pkl', help="参数保存文件名+pkl")


args_origin = parser.parse_args()

def args_wrapper(args):
    # 主要是对重复训练的保存路径进行封装
    # None
    args_origin.save_path = args_origin.source_path + '/' + args_origin.experiment_name + '/'
    # args_origin.save_path = args_origin.source_path + '/' 'your name' + '/' + args_origin.experiment_name + '/'
    if not os.path.exists(args_origin.save_path):
    	os.makedirs(args_origin.save_path)
    return args

args = args_wrapper(args_origin)

# todo: 打印时每行一个arg
=======
#!usr/bin/env python3
# -*- coding: utf-8 -*-
'''
定义和DQN算法相关的超参数
'''

import argparse
import os
import pprint

parser = argparse.ArgumentParser()

# runner
parser.add_argument("--env_name", type=str, default="airCombate", help="游戏名字") 
parser.add_argument("--flag_is_train", type=int, default="1", help="flag_IsTrain = 1, 一个训练，一个使用； flag_IsTrain = 0, 两个都是在使用")
parser.add_argument("--flag_focus_blue", type=int, default="1", help="flag_focus_blue = 1 时训练agent_blue； flag_focus_blue = 0 时训练agent_red")


# interactor
parser.add_argument("--episode", type=int, default="2000000", help="训练的最大episode数")
parser.add_argument("--store", type=int, default="10000", help="初始化经验池时运行的episode数")
parser.add_argument("--test_episode", type=int, default="100", help="测试时运行的episode数")
parser.add_argument("--train_episode", type=int, default="100", help="每训练多少个episode后启动测试")

# models
parser.add_argument("--net_frame", type=str, default='mlp', help="[mlp, cnn2mlp, cnn2rnn2mlp]")
parser.add_argument("--hidden_units", default=[128, 128], help="每个隐藏层的神经元数量")
parser.add_argument("--convs", default=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], help="每个卷积层设置")
parser.add_argument("--batch_size", type=int, default="256")
parser.add_argument("--gamma", type=float, default="0.99")
parser.add_argument("--learning_rate", type=float, default="0.00005", help="学习率")
parser.add_argument("--initial_epsilon", type=float, default="1.0", help="初始探索概率")
parser.add_argument("--decay_rate", type=float, default="1.0", help="探索率下降幅度")
parser.add_argument("--replay_size", type=int, default="100000")


# envs
parser.add_argument("--unit_type", type=str, default='default',help="[default,]")
parser.add_argument("--red_unit_type_list", default=['default'], help="飞机类型包括：[default,]")  # MvN环境中使用
parser.add_argument("--blue_unit_type_list", default=['default'], help="飞机类型包括：[default,]") # MvN环境中使用

parser.add_argument("--envs_type", type=str, default='2D_xy',help="场景类型：2D_xy, 2D_xz, 3D")

parser.add_argument("--map_area", type=int, default='3000',help="设定地图范围")
parser.add_argument("--map_scale", type=float, default='0.1',help="地图比例尺")
parser.add_argument("--map_t", type=float, default='0.5',help="时间间隔（步长，秒)")
parser.add_argument("--map_t_n", type=int, default='5',help="每个步长计算次数")
parser.add_argument("--env_random_seed", type=int, default='2',help="环境随机种子")

parser.add_argument("--init_scen", type=int, default='0',help="初始想定模式包括：0随机，1进攻，2防御，3同向，4中立")
parser.add_argument("--random_r", type=int, default='0',help="红方是否在初始化过程中随机")
parser.add_argument("--random_b", type=int, default='1',help="蓝方是否在初始化过程中随机")

parser.add_argument("--G", type=float, default='9.81',help="重力加速度")
parser.add_argument("--roll_rate", type=int, default='40',help="滚转角变化率")

parser.add_argument("--random_init_pos_b", type=int, default='1',help="[0 or 1]")
parser.add_argument("--random_init_pos_r", type=int, default='1',help="[0 or 1]")

parser.add_argument("--state_setting", type=str, default='orign_state',help="[orign_state, state_direct_pos]")

# unit
parser.add_argument("--Sum_Oil", type=int, default='1000',help="油量，即每个episode的最大step数量")

# utlis
parser.add_argument("--seed", type=int, default="125")

parser.add_argument("--source_path", type=str, default='../result',help="保存路径")                   
parser.add_argument("--experiment_name", type=str, default='blue_red_game', help="保存文件夹的名字")
parser.add_argument("--checkpoint_folder_name", type=str, default='_saved_networks/', help="参数保存文件夹的名字，加/结尾")
parser.add_argument("--file_name", type=str, default='_agent.pkl', help="参数保存文件名+pkl")


args_origin = parser.parse_args()

def args_wrapper(args):
    # 主要是对重复训练的保存路径进行封装
    # None
    args_origin.save_path = args_origin.source_path + '/' + args_origin.experiment_name + '/'
    # args_origin.save_path = args_origin.source_path + '/' 'your name' + '/' + args_origin.experiment_name + '/'
    if not os.path.exists(args_origin.save_path):
    	os.makedirs(args_origin.save_path)
    return args

args = args_wrapper(args_origin)

# todo: 打印时每行一个arg
>>>>>>> 79a627a30161c28eb9a9274ad8b37248f81f9a37
pprint.pprint(args)