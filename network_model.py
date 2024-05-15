# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/18 19:08
@Auth ： xiaolongtuan
@File ：network_model.py
"""
import copy
import math
import os
from datetime import datetime
import random

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from gen_topo_py import GenTopoPy


class NetworkModel():
    def __init__(self):
        self.G = nx.DiGraph()
        self.device_account = 0
        self.di_edge_account = 0

    def start_gen_topo_py(self):
        self.gen_topo_py = GenTopoPy()

    def get_topo_py(self, path):
        self.gen_topo_py.gen_file(path)

    def add_node(self, dpid, ip_add, mac_add):
        self.G.add_node(int(dpid), host_name=f's{dpid}', ip=ip_add, mac_address=mac_add)
        self.device_account += 1
        self.gen_topo_py.add_node(dpid)

    def add_double_edge(self, from_device_dpid, from_port, to_device_dpid, to_port):
        from_device_dpid = int(from_device_dpid)
        to_device_dpid = int(to_device_dpid)
        self.G.add_edge(from_device_dpid, to_device_dpid, from_port=from_port, to_port=to_port)
        self.G.add_edge(to_device_dpid, from_device_dpid, from_port=to_port, to_port=from_port)

        self.di_edge_account += 2
        self.gen_topo_py.add_link(from_device_dpid, from_port, to_device_dpid, to_port)

    def add_di_edge(self, from_device, from_port, to_device, to_port):
        self.G.add_edge(from_device, to_device, from_port=from_port, to_port=to_port)
        self.di_edge_account += 1

    def forward(self, cur_dpid, src, dis):  # cur != src != dis
        try:
            shortest_path = nx.dijkstra_path(self.G, source=src, target=dis, weight='weight')
            if cur_dpid in shortest_path:
                next = shortest_path[shortest_path.index(cur_dpid) + 1]
                target_edge = [(u, v, data) for u, v, data in self.G.edges(data=True) if u == cur_dpid and v == next]
                from_port = target_edge[0][2]['from_port']
                print(f"\nfrom {src} to {dis}, output from {cur_dpid}' port{from_port}")
                return from_port
            else:
                print(f"The path between {src} and {dis} don't through {cur_dpid}")
                return -1
        except nx.NetworkXNoPath:
            print(f"No path exists between {src} and {dis}")
            return -1

    def next_hop(self, cur, src, dis):  # cur != src != dis
        try:
            shortest_path = nx.dijkstra_path(self.G, source=src, target=dis, weight='weight')
            if cur in shortest_path:
                next = shortest_path[shortest_path.index(cur) + 1]
                print(f"\nfrom {src} to {dis}, from {cur} nexthop is {next['host_name']}")
                return next['host_name']
            else:
                print(f"The path between {src} and {dis} don't through {cur}")
                return -1
        except nx.NetworkXNoPath:
            print(f"No path exists between {src} and {dis}")
            return -1

    def random_weight(self, dataset_index):
        nodes = sorted(self.G.nodes())
        # 创建权重矩阵
        num_nodes = len(nodes)
        self.weight_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        for u, v, data in self.G.edges(data=True):
            weight_int = random.randint(1, self.device_account)  # 随机权重整数
            data['weight'] = weight_int
            i = nodes.index(u)
            j = nodes.index(v)
            self.weight_matrix[i][j] = weight_int

        folder_path = f"datas/weight_and_flow/{dataset_index}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.savetxt(os.path.join(folder_path, "weight_matrix.txt"), self.weight_matrix, fmt='%d', delimiter='\t')
        return folder_path

    def change_random_weight(self,dataset_index):
        nodes = sorted(self.G.nodes())
        # 创建权重矩阵
        num_nodes = len(nodes)
        self.weight_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        for u, v, data in self.G.edges(data=True):
            weight_int = random.randint(1, self.device_account)  # 随机权重整数
            data['weight'] = weight_int
            i = nodes.index(u)
            j = nodes.index(v)
            self.weight_matrix[i][j] = weight_int

        folder_path = f"datas/change_weight_and_flow/{dataset_index}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.savetxt(os.path.join(folder_path, "weight_matrix.txt"), self.weight_matrix, fmt='%d', delimiter='\t')

        # 更新权重矩阵
        self.new_G = copy.deepcopy(self.G)
        self.new_weight_matrix = copy.deepcopy(self.weight_matrix)
        num_edges = len(self.new_G.edges())
        change_edges = random.sample(list(self.new_G.edges(data=True)), math.ceil(num_edges * 0.1)) # 采样十分之一的边改变

        for u, v, data in change_edges:
            weight_int = random.randint(1, self.device_account)  # 随机权重整数
            data['weight'] = weight_int
            i = nodes.index(u)
            j = nodes.index(v)
            self.new_weight_matrix[i][j] = weight_int
        np.savetxt(os.path.join(folder_path, "changed_weight_matrix.txt"), self.weight_matrix, fmt='%d', delimiter='\t')
        return folder_path

    def print_network(self):
        print("Nodes:")
        for node, data in self.G.nodes(data=True):
            print(
                f"Node: {node}, IP: {data['ip']}, MAC Address: {data['mac_address']}")

        print("\nEdges:")
        for u, v, data in self.G.edges(data=True):
            print(
                f"Edge: {u}:port{data['from_port']} -> {v}:port{data['from_port']}, Weight: {data['weight']}")

    def draw_netgraph(self, path):
        pos = nx.spring_layout(self.G)

        # 绘制节点及节点属性标签
        node_labels = nx.get_node_attributes(self.G, 'host_name')
        nx.draw_networkx_nodes(self.G, pos, node_color='skyblue', node_size=1000)
        nx.draw_networkx_labels(self.G, pos, labels=node_labels)

        # 绘制边及边属性标签
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edges(self.G, pos, edge_color='gray', arrows=True)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)

        # 显示图形
        plt.axis('off')
        plt.savefig(path, format="PNG")
        plt.show()

    def load_net_graph(self, path):
        self.G = nx.read_graphml(path)
        self.device_account = len(self.G.nodes)
        self.di_edge_account = len(self.G.edges)
        # self.print_network()


if __name__ == '__main__':
    model = NetworkModel()
    model.start_gen_topo_py()
    # 添加节点和边
    model.add_node(dpid='1', ip_add='10.0.0.1', mac_add='00:00:00:00:00:01')
    model.add_node(dpid='2', ip_add='10.0.0.2', mac_add='00:00:00:00:00:02')
    model.add_node(dpid='3', ip_add='10.0.0.3', mac_add='00:00:00:00:00:03')
    model.add_node(dpid='4', ip_add='10.0.0.4', mac_add='00:00:00:00:00:04')
    model.add_node(dpid='5', ip_add='10.0.0.5', mac_add='00:00:00:00:00:05')
    model.add_node(dpid='6', ip_add='10.0.0.6', mac_add='00:00:00:00:00:06')
    model.add_node(dpid='7', ip_add='10.0.0.7', mac_add='00:00:00:00:00:07')
    model.add_node(dpid='8', ip_add='10.0.0.8', mac_add='00:00:00:00:00:08')
    model.add_node(dpid='9', ip_add='10.0.0.9', mac_add='00:00:00:00:00:09')
    model.add_node(dpid='10', ip_add='10.0.0.10', mac_add='00:00:00:00:00:10')
    model.add_node(dpid='11', ip_add='10.0.0.11', mac_add='00:00:00:00:00:11')
    model.add_node(dpid='12', ip_add='10.0.0.12', mac_add='00:00:00:00:00:12')
    model.add_node(dpid='13', ip_add='10.0.0.13', mac_add='00:00:00:00:00:13')
    model.add_node(dpid='14', ip_add='10.0.0.14', mac_add='00:00:00:00:00:14')

    model.add_double_edge(from_device_dpid='1', from_port=0, to_device_dpid='3', to_port=0)
    model.add_double_edge(from_device_dpid='1', from_port=1, to_device_dpid='2', to_port=0)
    model.add_double_edge(from_device_dpid='1', from_port=2, to_device_dpid='6', to_port=0)
    model.add_double_edge(from_device_dpid='2', from_port=1, to_device_dpid='3', to_port=1)
    model.add_double_edge(from_device_dpid='2', from_port=2, to_device_dpid='4', to_port=0)
    model.add_double_edge(from_device_dpid='3', from_port=2, to_device_dpid='9', to_port=0)
    model.add_double_edge(from_device_dpid='4', from_port=1, to_device_dpid='7', to_port=0)
    model.add_double_edge(from_device_dpid='4', from_port=2, to_device_dpid='14', to_port=0)
    model.add_double_edge(from_device_dpid='4', from_port=3, to_device_dpid='5', to_port=0)
    model.add_double_edge(from_device_dpid='6', from_port=1, to_device_dpid='11', to_port=0)
    model.add_double_edge(from_device_dpid='6', from_port=2, to_device_dpid='7', to_port=1)
    model.add_double_edge(from_device_dpid='7', from_port=2, to_device_dpid='8', to_port=0)
    model.add_double_edge(from_device_dpid='8', from_port=1, to_device_dpid='9', to_port=1)
    model.add_double_edge(from_device_dpid='9', from_port=2, to_device_dpid='10', to_port=0)
    model.add_double_edge(from_device_dpid='5', from_port=1, to_device_dpid='10', to_port=1)
    model.add_double_edge(from_device_dpid='10', from_port=2, to_device_dpid='12', to_port=0)
    model.add_double_edge(from_device_dpid='10', from_port=3, to_device_dpid='13', to_port=0)
    model.add_double_edge(from_device_dpid='11', from_port=1, to_device_dpid='12', to_port=1)
    model.add_double_edge(from_device_dpid='11', from_port=2, to_device_dpid='13', to_port=1)
    model.add_double_edge(from_device_dpid='12', from_port=2, to_device_dpid='14', to_port=1)
    model.add_double_edge(from_device_dpid='13', from_port=2, to_device_dpid='14', to_port=2)
    model.get_topo_py("datas/NSFnet.py")
    # model.random_weight(22)
    model.draw_netgraph('datas/nsfnet_topo.png')
    model.forward(4, 1, 7)
    nx.write_graphml(model.G, "datas/topology.graphml")
