# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/4 14:17
@Auth ： xiaolongtuan
@File ：draw.py
"""
import pickle
import matplotlib.pyplot as plt

net = 'SAGE'

# 加载数据
with open(f'res/{net}_train_loss.pkl', 'rb') as f:
    train_loss = pickle.load(f)
with open(f'res/{net}_test_loss.pkl', 'rb') as f:
    test_loss = pickle.load(f)
with open(f'res/{net}_test_accuracy.pkl', 'rb') as f:
    test_accuracy = pickle.load(f)

# 绘制曲线图
def draw(data,name):
    x = list(range(len(data)))
    # 绘制曲线图
    plt.plot(x, data, label=name)
    plt.xlabel('epcho')
    plt.ylabel(name)
    plt.title(name)
    plt.legend()
    plt.show()
draw(train_loss,'train_loss')
draw(test_loss,'test_loss')
draw(test_accuracy,'test_accuracy')
