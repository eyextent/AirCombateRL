#!usr/bin/env python3
# -*- coding: utf-8 -*-
'''
定义和策略梯度方法相关的超参数
'''
import argparse

parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument()




args_origin = parser.parse_args()

def args_wrapper(args):
    # 主要是对重复训练的保存路径进行封装
    # None
    return args

args = args_wrapper(args_origin)

# todo: 打印时每行一个arg
print(args)
