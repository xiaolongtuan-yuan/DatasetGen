# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/9 15:09
@Auth ： xiaolongtuan
@File ：test.py
"""
import numpy as np

# 创建一个示例数组
arr = np.array([[1, 2, 3], [4, 5, 6]]).astype(str)

# 将数组转换为字符串
table_str = '\n'.join([' '.join(line) for line in arr])

print(table_str)
