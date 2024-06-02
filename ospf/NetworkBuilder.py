# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/10 10:55
@Auth ： xiaolongtuan
@File ：build_net.py
"""
import copy
import json
import math
import os
import pickle
import random
from scipy.spatial import Delaunay
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np

from ospf.Device import *
import networkx as nx
from networkx import spring_layout
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class QueryType(Enum):
    ROUTE = 1
    INTERFACE = 2
    NETWORK = 3


class IpDistributor:
    '''
    全部使用C类网络
    '''

    def __init__(self):
        self.counter = 0
        self.counter2 = 0
        self.base_add = ipaddress.IPv4Address("1.0.0.0")
        self.mask = ipaddress.IPv4Address("255.255.255.0")

        self.loop_add = ipaddress.IPv4Address("1.1.1.1")
        self.loop_mask = ipaddress.IPv4Address("255.255.255.255")

    def assign_address(self, interface_type: Interface_type):
        if interface_type == Interface_type.GigabitEthernet:
            network_add = ipaddress.IPv4Address((int(self.base_add) + self.counter * 256))
            self.counter += 1
            return (network_add, self.mask)
        elif interface_type == Interface_type.Loopback:
            network_add = ipaddress.IPv4Address((int(self.loop_add) + self.counter2))
            self.counter2 += 1
            return (network_add, self.loop_mask)
        else:
            raise Exception("分配接口网络类型不存在")


class NetworkBuilder:
    def __init__(self, device_account, network_root, need_embeded=False, is_Di=True):
        self.device_account = device_account
        self.network_root = network_root

        path_items = network_root.split('/')
        self.dataset_dir = path_items[0]
        self.net_scale = path_items[-2]
        self.net_seq = path_items[-1]

        config_path = f'{self.network_root}/configs'

        if not os.path.exists(self.network_root):
            os.makedirs(self.network_root)
            os.makedirs(config_path)

        if not os.path.exists(os.path.join(self.dataset_dir, "omnet_file", self.net_scale, "networks")):
            os.makedirs(os.path.join(self.dataset_dir, "omnet_file", self.net_scale, "networks"))
            os.makedirs(os.path.join(self.dataset_dir, "omnet_file", self.net_scale, "ini_dir"))
        self.ip_distributor = IpDistributor()
        self.need_embeded = need_embeded
        self.max_weight = 1000
        self.weight_matrix = None
        self.is_Di = is_Di  # 有向图
        self.channel_num = 0
        self.node_dst_map = {}  # 每个节点只向一个目的地发包

    def random_planar_graph(self, num_nodes, MAX_INTERFACE=10):
        random_state = np.random.RandomState()
        pos = random_state.rand(num_nodes, 2)
        simps = Delaunay(pos).simplices
        G = nx.DiGraph()

        for i in range(num_nodes):
            G.add_node(i)

        def add_edge(src, dst):
            if G.out_degree(src) < MAX_INTERFACE and G.out_degree(dst) < MAX_INTERFACE:
                G.add_edge(src, dst)
                G.add_edge(dst, src)

        for tri in simps:
            add_edge(tri[0], tri[1])
            add_edge(tri[1], tri[2])
            add_edge(tri[2], tri[0])

        return G

    def build_graph(self):  # 节点从0开始
        # self.G = nx.DiGraph() if self.is_Di else nx.Graph()  # 有向图
        #
        # for i in range(self.device_account):
        #     host_name = f'R{i}'
        #     device = Device(device_type=Device_type.ROUTER, host_name=host_name)
        #     self.G.add_node(i, device=device)
        #
        # temp_G = copy.deepcopy(self.G)
        # edge_probability = 0.1
        # while len(list(nx.weakly_connected_components(temp_G) if self.is_Di else nx.connected_components(
        #         temp_G))) > 1:  # 避免出现新孤岛
        #     temp_G = copy.deepcopy(self.G)
        #     for i in range(self.device_account):
        #         for j in range(i + 1, self.device_account):  # 避免自环和重复边
        #             if random.random() < edge_probability:  # 控制边的概率
        #                 temp_G.add_edge(i, j)
        #                 if self.is_Di:
        #                     temp_G.add_edge(j, i)
        # self.G = copy.deepcopy(temp_G)
        self.G = self.random_planar_graph(self.device_account)

        for node in self.G.nodes():
            host_name = f'R{node}'
            device = Device(device_type=Device_type.ROUTER, host_name=host_name)
            self.G.nodes[node]['device'] = device
        self.random_weight()
        self.address_distribution(name='weight_matrix')

    def draw_graph(self):
        pos = spring_layout(self.G)
        fig, ax = plt.subplots()
        node_labels = {node: self.G.nodes[node]['device'].host_name for node in self.G.nodes()}

        nx.draw(self.G, pos=pos, with_labels=True, node_size=400, node_color='lightblue',
                font_size=12, font_weight='bold',
                arrows=True, ax=ax)
        label_pos = {k: (v[0], v[1] + 0.1) for k, v in pos.items()}
        nx.draw_networkx_labels(self.G, pos=label_pos, labels=node_labels)
        plt.savefig(os.path.join(self.network_root, "topo.png"), format='png')
        # plt.show()
        plt.clf()

    def draw_dev_graph(self, pos=None, figsize=(10, 10), label="", use_node_labels=True):
        G = self.G.copy()

        pos = pos if pos is not None else nx.drawing.layout.spring_layout(G, weight=None)

        node_color = ["lightblue"] * len(G.nodes())

        interface_pos = {}
        interface_labe = {}
        for edge in G.edges():
            u, v = edge
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            out_pos = (x1 + (x2 - x1) / 6, y1 + (y2 - y1) / 6)

            interface_labe[edge] = G.edges[u, v]['interface_info']['out']
            interface_pos[edge] = out_pos

        fig = plt.Figure(figsize=figsize)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        plt.axis('off')

        def node_label(n):
            return G.nodes[n]['device'].host_name

        nx.draw_networkx_nodes(G, pos, node_color=node_color, ax=ax)
        nx.draw_networkx_edge_labels(G, label_pos=0.8, pos=pos, edge_labels=interface_labe, ax=ax)
        nx.draw_networkx_labels(G, labels=dict([(n, node_label(n)) for n in G.nodes()]), pos=pos, ax=ax)
        filtered_edges = [(u, v) for u, v, d in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, edgelist=filtered_edges, ax=ax)

        ax.set_title(label)
        canvas.print_figure(f'{self.network_root}/network_inf_graph.png', dpi=300)

    def address_distribution(self, name):
        '''
        为所有设备的物理接口和逻辑接口分配网络
        :return:
        '''
        # self.weight_matrix = np.zeros((self.device_account, self.device_account), dtype=int)
        num_nodes = len(self.G.nodes())
        visited = np.zeros((num_nodes, num_nodes), dtype=bool)
        for node1, node2, data in self.G.edges(data=True):  # 对物理接口进行赋值
            if visited[node1][node2]: continue
            prefix, mask = self.ip_distributor.assign_address(Interface_type.GigabitEthernet)
            ip1 = str(ipaddress.IPv4Address(int(prefix) + 1))
            ip2 = str(ipaddress.IPv4Address(int(prefix) + 2))

            device1 = self.G.nodes[node1]['device']
            weight1 = self.weight_matrix[node1][node2]
            interface_index1 = device1.add_interface(interface_type=Interface_type.GigabitEthernet, network_add=ip1,
                                                     prefix=str(prefix),
                                                     mask=mask, weight=weight1)

            device2 = self.G.nodes[node2]['device']
            weight2 = self.weight_matrix[node2][node1]
            interface_index2 = device2.add_interface(interface_type=Interface_type.GigabitEthernet, network_add=ip2,
                                                     prefix=str(prefix),
                                                     mask=mask, weight=weight2)
            if self.is_Di:
                data['weight'] = weight1
                self.G[node2][node1]['weight'] = weight2
            else:
                data['weight'] = weight1 + weight2
                self.G[node2][node1]['weight'] = weight1 + weight2

            self.G[node1][node2]['interface_info'] = {'out': interface_index1, 'in': interface_index2}  # 记录边连接的端口
            self.G[node2][node1]['interface_info'] = {'out': interface_index2, 'in': interface_index1}
            visited[node1][node2] = True
            visited[node2][node1] = True

        for node in self.G.nodes:  # 对逻辑接口进行赋值
            prefix, mask = self.ip_distributor.assign_address(Interface_type.Loopback)
            device = self.G.nodes[node]['device']
            device.add_interface(interface_type=Interface_type.Loopback, network_add=str(prefix), prefix=str(prefix),
                                 mask=mask, weight=0)

        # np.savetxt(os.path.join(self.network_root, "weight_matrix.pkl"), self.weight_matrix, fmt='%d', delimiter='\t')
        # with open(os.path.join(self.network_root, f"{name}.pkl"), 'wb') as file:
        #     pickle.dump(self.weight_matrix, file)

    def gen_ospf_config(self):
        for node in self.G.nodes:
            device = self.G.nodes[node]['device']
            device.make_ospf_config_file(self.network_root, self.need_embeded)
        # 随机找一个节点作为终点，不用这种方法了
        # for j in self.G.nodes():
        #     dst = j
        #     while dst == j:
        #         dst = random.sample(list(self.G.nodes), 1)[0]
        #     self.node_dst_map[j] = dst

    def channel_register(self):
        delay_leval = [['1ms', '10ms', '20ms'], ['40ms', '80ms', '150ms'], ['200ms', '400ms', '1s']]  # 低，中，高延迟
        data_rate_leval = [['20kbps', '100kbps', '500kbps'], ['1Mbps', '5Mbps', '25Mbps'],
                           ['100Mbps', '400Mbps', '1Gbps']]  # 低，中，高速
        per_leval = [['0.001', '0.004', '0.08'], ['0.01', '0.04', '0.08'], ['0.1', '0.2', '0.4']]
        random_choose = [delay_leval, data_rate_leval, per_leval]
        with open("ospf/resource/edge_attribute_tem.txt", 'r') as file:
            channel_attribute_tem = file.read()

        channel_name = f"C{self.channel_num}"
        channel_value = []
        for data_levals in random_choose:
            leval = random.choice(data_levals)
            value = random.choice(leval)
            channel_value.append(value)
        channel_attribute_str = channel_attribute_tem.format(channel_name, channel_value[0], channel_value[1],
                                                             channel_value[2])
        self.channel_num += 1
        channel_attribute = {
            'delay': channel_value[0],
            'datarate': channel_value[1],
            'per': channel_value[2]
        }
        return channel_name, channel_attribute_str, channel_attribute

    def gen_ned_file(self):
        edge_tem = 'rte[{}].port[{}] <--> {} <--> rte[{}].port[{}];'  # 中间的channel要大变
        with open("ospf/resource/ned_template.txt", 'r') as ned_file:
            ned_template = ned_file.read()

        channel_str_list = []  # 初始化channel
        edge_str_list = []  # 边连接，携带channel变量
        num_nodes = len(self.G.nodes())
        visited = np.zeros((num_nodes, num_nodes), dtype=bool)
        columns = ['node1', 'node2', 'delay', 'datarate', 'per']
        edge_df = pd.DataFrame(columns=columns)
        for node1, node2, data in self.G.edges(data=True):  # 遍历物理边
            if visited[node1][node2]: continue
            channel_name, channel_attribute_str, channel_attribute = self.channel_register()
            channel_str_list.append(channel_attribute_str)
            edge_str_list.append(
                edge_tem.format(node1, data['interface_info']['out'], channel_name, node2,
                                data['interface_info']['in']))
            channel_attribute['node1'] = node1
            channel_attribute['node2'] = node2
            edge_df.loc[len(edge_df)] = channel_attribute
            visited[node2][node1] = True
        # channel 注册
        channel_str = "\n        ".join(channel_str_list)
        # 边
        edge_str = '\n        '.join(edge_str_list)
        ned_str = ned_template.format(str(self.net_seq), channel_str, str(num_nodes), edge_str)

        os.makedirs(os.path.join("omnet_file", "networks"), exist_ok=True)
        with open(os.path.join(self.dataset_dir, "omnet_file", self.net_scale, "networks",
                               f"Myned{str(self.net_seq)}.ned"), 'w') as f:
            f.write(ned_str)

        edge_df.to_csv(os.path.join(self.network_root, "topology.csv"), index=False)

    def get_route_table(self):
        route_table = np.full((len(self.G.nodes), len(self.G.nodes)), -1)

        for source in self.G.nodes():

            for target in self.G.nodes():
                if source != target:
                    try:
                        path = nx.dijkstra_path(self.G, source, target)
                        next_node = path[1]
                        route_table[source][target] = self.G[source][next_node]['interface_info']['out']
                    except nx.NetworkXNoPath:
                        continue
        np.savetxt(os.path.join(self.network_root, 'route_table.txt'), route_table, fmt='%d')
        return route_table

    def gen_ini_file(self):
        route_tabel = self.get_route_table().astype(str)
        table_str = '\\n'.join([' '.join(line) for line in route_tabel])
        with open("ospf/resource/ini_template.txt", 'r') as ini_file:
            ini_format = ini_file.read()

        node_info = []
        for n in self.G.nodes:
            node_info.append(f"Myned{str(self.net_seq)}.rte[{n}].appType = \"router\"")
            node_info.append(f"Myned{str(self.net_seq)}.rte[{n}].address = {n}")
            node_info.append(f"Myned{str(self.net_seq)}.rte[{n}].destAddresses = \"{self.node_dst_map[n]}\"")

        node_type_str = '\n'.join(node_info)
        net_node = [str(n) for n in self.G.nodes]
        dest_str = ' '.join(net_node)
        ini_str = ini_format.format(str(self.net_seq), str(self.net_seq), str(self.net_seq), table_str,
                                    str(self.net_seq), str(self.net_seq), node_type_str)

        os.makedirs(os.path.join("omnet_file", "ini_dir"), exist_ok=True)
        with open(os.path.join(self.dataset_dir, "omnet_file", self.net_scale, "ini_dir",
                               f"omnetpp{str(self.net_seq)}.ini"),
                  "w") as ini_file:
            ini_file.write(ini_str)

    def get_edges(self):
        edge_list = []
        for edge in self.G.edges():
            device1 = self.G.nodes[edge[0]]['device']
            device2 = self.G.nodes[edge[1]]['device']
            edge_list.append(f"[{device1.host_name},{device2.host_name}]")
        return ",".join(edge_list)

    def save_pyg(self):
        edge_index = torch.tensor(list(self.G.edges())).t().contiguous()
        x = torch.arange(self.G.number_of_nodes()).view(-1, 1).float()  # 假设节点特征为节点编号
        data = Data(x=x, edge_index=edge_index)
        torch.save(data, os.path.join(str(self.network_root), 'graph_edges.pt'))

    def get_path_from_start(self, start):
        while True:
            dest = random.sample(list(self.G.nodes), 1)[0]
            if not start == dest:
                if not self.G.has_edge(start, dest):  # 不是邻居节点
                    break
        path = nx.dijkstra_path(self.G, source=start, target=dest, weight='weight')
        return dest, path


    def get_path(self, start, dst):
        path = nx.dijkstra_path(self.G, source=start, target=dst, weight='weight')
        return path

    def record_edges(self, begin, edge_record: []):  # 用于将小图拼接为大图时，构造data.edge_index，向edge_record添加边
        edges = self.G.edges()
        for start, end in edges:
            edge_record[0].append(begin * 20 + start)
            edge_record[1].append(begin * 20 + end)

    def save_pkl(self):
        edges = self.G.edges()
        n = len(edges)
        start_nodes = []
        end_nodes = []

        for start, end in edges:
            start_nodes.append(start)
            end_nodes.append(end)

        edges = [start_nodes, end_nodes]
        # print(edges)

        with open(f'{self.network_root}/graph_edges.pkl', 'wb') as f:
            pickle.dump(edges, f)

        with open(f'{self.network_root}/net_model.pkl', 'wb') as f:
            pickle.dump(self.G, f)

    def load_pyg(self):
        data = torch.load(os.path.join(str(self.network_root), 'graph.pt'))
        # print(data.edge_index)

    def get_node_presentation(self, node):
        node_presentation = []
        connection_form = f"this {self.G.nodes[node]['device'].host_name}: the weights of the edges connecting this node to its neighbor nodes are as follows："
        edge_form = "to node {}, the weight is {}"
        node_presentation.append(connection_form)
        for neighbor in self.G.neighbors(node):
            node_presentation.append(edge_form.format(self.G.nodes[neighbor]['device'].host_name,
                                                      self.weight_matrix[node][neighbor]))
        return "\n".join(node_presentation)

    def get_node_presentation_type2(self, node):
        node_presentation = []
        connection_form = f"this {self.G.nodes[node]['device'].host_name}: the shortest forwarding path from this node to other nodes in the network are as follows："
        path_format = "to node {}, the path is {}"
        other_nodes = [i for i in self.G.nodes() if i != node]
        node_presentation.append(connection_form)
        for i in other_nodes:
            path = nx.dijkstra_path(self.G, node, i, weight='weight')
            path_str = ' -> '.join([self.G.nodes[j]['device'].host_name for j in path])
            node_presentation.append(path_format.format(self.G.nodes[i]['device'].host_name, path_str))
        return "\n".join(node_presentation)

    def get_query_and_answer(self, cur_node, query_type: QueryType):
        if query_type == QueryType.ROUTE.value:
            dest, path = self.get_path_from_start(cur_node)
            query = f"What are the shortest OSPF paths from {self.G.nodes[cur_node]['device'].host_name} and {self.G.nodes[dest]['device'].host_name}"
            hostname_path = [self.G.nodes[node]['device'].host_name for node in path]
            answer = ' - '.join(hostname_path)
            return query, answer, json.dumps(hostname_path)
        elif query_type == QueryType.INTERFACE.value:
            this_device = self.G.nodes[cur_node]['device']
            query = f"List all the interface info of the {this_device.host_name} device"
            answer_list = []
            label_list = []
            for interface in this_device.interface_list:
                inter_str = f"{interface.get_interface_name()}:Ipv4 address is {interface.address}，mask is {interface.mask}，OSPF cost is {interface.weight}"
                answer_list.append(inter_str)
                label_list.append({
                    'interface': interface.get_interface_name(),
                    'address': str(interface.address),
                    'mask': str(interface.mask)
                })
            answer = '\n'.join(answer_list)
            return query, answer, json.dumps(label_list)
        elif query_type == QueryType.NETWORK.value:
            this_device = self.G.nodes[cur_node]['device']
            query = f"List all the Ipv4 network of the {this_device.host_name} device"
            answer_list = []
            label_list = []
            answer_list.append(f"there are {len(this_device.networks)} networks:")
            for network in this_device.networks:
                answer_list.append(f"{network['prefix']} {network['mask']}")
                label_list.append({
                    'prefix': str(network['prefix']),
                    'mask': str(network['mask'])
                })
            answer = '\n'.join(answer_list)
            return query, answer, json.dumps(label_list)
        else:
            raise Exception("不存在的query_type!")

    def random_weight(self):
        # 随机权重矩阵,只更新0.05边
        if self.weight_matrix is None:
            self.weight_matrix = np.full((self.device_account, self.device_account), np.inf)
            for u, v, data in self.G.edges(data=True):
                weight_int = random.randint(1, self.max_weight)  # 随机权重整数
                data['weight'] = weight_int
                self.weight_matrix[u][v] = weight_int
            return self.weight_matrix

        num_edges = len(self.G.edges())
        old_route_table = self.get_route_table()
        while True:
            change_edges = random.sample(list(self.G.edges(data=True)), math.ceil(num_edges * 0.05))  # 采样20分之一的边改变
            for u, v, data in change_edges:
                weight_int = random.randint(1, self.max_weight)  # 随机权重整数
                data['weight'] = weight_int
                self.weight_matrix[u][v] = weight_int
            new_route_table = self.get_route_table()
            if not np.array_equal(old_route_table, new_route_table):  # 如果转发平面未发生变化，则重新改变权重
                break

        path_changed_node_pairs = {}
        for i in range(old_route_table.shape[0]):  # 起始节点
            path_changed_node_pairs[i] = []
            for j in range(old_route_table.shape[1]):
                if not old_route_table[i, j] == new_route_table[i, j]:
                    path_changed_node_pairs[i].append(j)
            if len(path_changed_node_pairs[i]) == 0: # 随机选择目的地节点，切changed=0
                dst = i
                while dst == i:
                    dst = random.sample(list(self.G.nodes), 1)[0]
                self.node_dst_map[i] = dst
            else:
                dst = random.sample(path_changed_node_pairs[i], 1)[0]
                self.node_dst_map[i] = dst

        return self.weight_matrix

    def path_record(self):
        n = len(self.G.nodes())
        graph_path = [[0 for _ in range(n)] for _ in range(n)]
        for i in self.G.nodes():
            for j in self.G.nodes():
                if not i == j:
                    path = self.get_path(i,j)
                    graph_path[i][j] = path
        return graph_path


if __name__ == '__main__':
    builder = NetworkBuilder(10, 0)
    builder.build_graph()
    builder.draw_graph()
    builder.gen_ospf_config()
    builder.save_pkl()
    # builder.load_pyg()
    builder.save_pkl()
