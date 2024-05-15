# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/21 21:35
@Auth ： xiaolongtuan
@File ：easy_simulation.py
"""
import json
import os
import random

import networkx as nx
import csv

import numpy as np

from network_model import NetworkModel

net_model = NetworkModel()
net_model.load_net_graph("datas/topology.graphml")

start = 1
end = 11
for i in range(start, end):
    folder_path = net_model.random_weight(i)
    forwarding_table = []  # 存放转发表信息

    # 抽样不可达的节点对：
    num_pairs = random.randint(0, net_model.G.number_of_edges())
    # 随机抽样节点对
    random_pairs = random.choices(list(net_model.G.nodes), k=num_pairs * 2)
    # 将抽样的节点对转换为元组列表
    node_pairs = [(random_pairs[i], random_pairs[i + 1]) for i in range(0, len(random_pairs), 2)  if not random_pairs[i] == random_pairs[i + 1]]

    reachable_matrix = np.zeros((len(net_model.G.nodes), len(net_model.G.nodes)), dtype=int)
    for pair in node_pairs:
        reachable_matrix[int(pair[0])-1][int(pair[1])-1] = 1
    np.savetxt(os.path.join(folder_path, "reachable_matrix.txt"), reachable_matrix, fmt='%d', delimiter='\t')

    for node1 in net_model.G.nodes:
        for node2 in net_model.G.nodes:
            if (node1, node2) in node_pairs:  # 不可达，洗发drop
                forwarding_table.append({
                    'datapath_id': node1,
                    'Head Fields': {
                        'src_ipv4': net_model.G.nodes[node1]['ip'],
                        'dis_ipv4': net_model.G.nodes[node2]['ip']
                    },
                    'input_port': '',
                    'action': 'drop'
                })
                continue

            if node1 == node2:
                continue

            path = nx.dijkstra_path(net_model.G, source=node1, target=node2, weight='weight')
            # todo 比较转发平面差异

            port_in = ''
            for i in range(len(path) - 1):
                cur = path[i]
                next_node = path[i + 1]
                port_out = net_model.G[cur][next_node]['from_port'] if 'from_port' in net_model.G[cur][
                    next_node] else ""
                # 将信息写入转发表
                forwarding_table.append({
                    'datapath_id': cur,
                    'Head Fields': {
                        'src_ipv4': net_model.G.nodes[node1]['ip'],
                        'dis_ipv4': net_model.G.nodes[node2]['ip']
                    },
                    'input_port': port_in,
                    'action': 'ouput',
                    'out_port': port_out
                })
                port_in = net_model.G[cur][next_node]['to_port'] if 'to_port' in net_model.G[cur][next_node] else ""

    sorted(forwarding_table, key=lambda entry: entry['datapath_id'])
    # 显示转发表信息
    json_str = json.dumps(forwarding_table, ensure_ascii=False)
    with open(os.path.join(folder_path, "forwarding_table.json"), "w") as file:
        file.write(json_str)
