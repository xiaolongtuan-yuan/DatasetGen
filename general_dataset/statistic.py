# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/5 10:53
@Auth ： xiaolongtuan
@File ：statistic.py
"""
import os
import json


def read_json_files_in_directory(directory, datalist:[]):
    # 遍历目录中的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否为.json
            if file.endswith('.json'):
                # 构建文件的完整路径
                file_path = os.path.join(root, file)

                # 尝试打开并读取JSON文件
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        print(f"文件名: {file}, 内容: {json_data}")
                        datalist.append({"file": file,"json_data":json_data})
                except FileNotFoundError:
                    print(f"文件未找到: {file_path}")
                except json.JSONDecodeError as e:
                    print(f"JSON解码错误: {file_path}, 错误信息: {e}")

                # 调用函数，传入你想要搜索的目录

datalist = []
read_json_files_in_directory('./',datalist)
total_len = 0
for i in datalist:
    total_len += len(i["json_data"])
print(total_len)
