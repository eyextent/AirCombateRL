#!usr/bin/env python3
# -*- coding: utf-8 -*-

'''
结果保存与展示相关
'''

'''
基本功能：
【1】存储
	创建字典
	存储时采用 关键字方式 进行存储，即 name_1=xxx
		若 字典的kyes 中不包含name_1，则 创建 列表。创建之后的字典即 {name_1:[]}
		若 字典的keys 中包含name_1，则 直接采用append进行添加

	# 思考：是否需要添加 iteration

【2】文件读写
	将存储的数据写入本地文件(json，xls 均可）

'''
import json
import pandas as pd
import numpy as np
from _collections import defaultdict


class Logger():
    def __init__(self):
        self.res_dict = defaultdict(list)

    #
    #    def create_file(self,filename):
    ##         """
    ##         创建日志文件夹和日志文件
    ##         :param filename:
    ##         :return:
    #         path = filename[0:filename.rfind("/")]
    #         if not os.path.isdir(path):  # 无文件夹时创建
    #             os.makedirs(path)
    #         if not os.path.isfile(filename):  # 无文件时创建
    #             fd = open(filename, mode="w", encoding="utf-8")
    #             fd.close()
    #         else:
    #             pass

    # 存储
    def store(self, ks, vs):
        '''
        使用defaultdict，对于不存在的key，生成默认值list
        '''
        self.res_dict[ks].append(vs)

    # 编码为json格式
    def dump_fun(self, dump_file):
        with open(dump_file, 'w') as f:
            json.dump(self.res_dict, f, sort_keys=False, indent=4, separators=(',', ': '), cls=NpEncoder)

    # 解码json存储到文件
    def load_fun(self, load_file):
        with open(load_file, 'r') as f:
            result = json.load(f)
        return result

    # 可变参数的存储
    def print_console(self, *args, **kwargs):
        if args == ():
            print('Experiment Data = ', kwargs)
        else:
            print('Experiment Data = ', kwargs)
            print('Action data = ', args)

    # 将json文件转为csv文件
    def json_to_csv(self, json_file, csv_file,name):
        # 分别读，创建文件
        json_fp = open(json_file, 'r', encoding='utf8')
        csv_fp = open(csv_file, 'w', newline='')

        # 提取表头和表的内容
        data_list = json.load(json_fp)

        k_list = []
        for k in data_list.keys():
            k_list.append(k)

        v_list = []
        for v in data_list.values():
            v_list.append(v)

        frames = []
        for i in range(len(k_list)):
            res_se = pd.Series(v_list[i], name=k_list[i])
            frames.append(res_se)
        df_data = pd.concat(frames, axis=1)
        df_data.to_csv(csv_file,index=True,index_label=name)  # 默认生成序号,修改列名

        json_fp.close()
        csv_fp.close()

    # 将多个csv文件整合为一个
    def concat_csv(self, input_file1, input_file2, concat_file):
        df1 = pd.read_csv(input_file1, encoding='gbk')
        df2 = pd.read_csv(input_file2, encoding='gbk')

        output_file = pd.concat([df1, df2], axis=1)
        output_file.to_csv(concat_file, index=False, encoding='gbk')


# 序列化自定义的类
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
