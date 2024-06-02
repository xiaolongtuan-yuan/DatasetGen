# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/24 14:12
@Auth ： xiaolongtuan
@File ：add_qos_data.py
"""
import os
import re

import pandas as pd


def get_sub_csv(dir_path):
    file_dict = {}
    for filename in os.listdir(dir_path):
        # 检查文件扩展名是否为.csv
        if filename.endswith('.csv'):
            net_id = int(re.findall(r'\d+', filename)[0])
            # 构造完整的文件路径
            file_path = os.path.join(dir_path, filename)
            file_dict[net_id] = file_path
    return file_dict


def get_sub_dataset(dir_path):
    file_dict = {}
    for filename in os.listdir(dir_path):
        if filename.isdigit():
            # 构造完整的文件路径
            file_path = os.path.join(dir_path, filename, 'dataset.jsonl')
            file_dict[int(filename)] = file_path
    return file_dict

net_scale = 'm'
print(net_scale)
csv_files = get_sub_csv(f'net_path_df/{net_scale}')
dataset_files = get_sub_dataset(net_scale)

for netid, csv_file in csv_files.items():
    dataset_file = dataset_files[netid]

    dataset_df = pd.read_json(dataset_file, lines=True)
    dataset_df = dataset_df.rename(columns={'node_ids': 'src'})

    qos_df = pd.read_csv(csv_file)
    qos_df['pkgLossRate'] = 1 - qos_df['pkgLossRate']
    new_df = pd.merge(dataset_df, qos_df, on=['src', 'dst'])
    new_df.to_json(dataset_file, orient="records", lines=True)
