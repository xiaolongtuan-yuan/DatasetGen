# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/9 15:11
@Auth ： xiaolongtuan
@File ：statistic_config.py
"""
import os

def count_characters_in_folder(folder_path):
    total_characters = 0

    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # 仅统计文本文件的字符数
            if os.path.isfile(file_path) and file_path.endswith('.cfg'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    # 读取文件内容并统计字符数
                    content = f.read()
                    total_characters += len(content)

    return total_characters

# 指定要统计字符数的文件夹路径
folder_path = 'ospf_dataset/0/configs'
total_characters = count_characters_in_folder(folder_path)
print("Total characters in folder:", total_characters)
