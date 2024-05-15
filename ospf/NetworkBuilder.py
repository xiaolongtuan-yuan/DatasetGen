# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/10 10:55
@Auth ： xiaolongtuan
@File ：build_net.py
"""
import copy
import json
import math
import pickle
import random
import torch
from torch_geometric.data import Data
import numpy as np

from ospf.Device import *
import networkx as nx
from networkx import spring_layout
import matplotlib.pyplot as plt

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
        config_path = f'{self.network_root}/configs'

        if not os.path.exists(self.network_root):
            os.makedirs(self.network_root)
            os.makedirs(config_path)
        self.ip_distributor = IpDistributor()
        self.need_embeded = need_embeded
        self.max_weight = 1000
        self.weight_matrix = None
        self.is_Di = is_Di  # 有向图

    def build_graph(self):  # 节点从0开始
        self.G = nx.DiGraph() if self.is_Di else nx.Graph()  # 有向图

        for i in range(self.device_account):
            host_name = f'R{i}'
            device = Device(device_type=Device_type.ROUTER, host_name=host_name)
            self.G.add_node(i, device=device)

        temp_G = copy.deepcopy(self.G)
        edge_probability = 0.1
        while len(list(nx.weakly_connected_components(temp_G) if self.is_Di else nx.connected_components(
                temp_G))) > 1:  # 避免出现新孤岛
            temp_G = copy.deepcopy(self.G)
            for i in range(self.device_account):
                for j in range(i + 1, self.device_account):  # 避免自环和重复边
                    if random.random() < edge_probability:  # 控制边的概率
                        temp_G.add_edge(i, j)
                        if self.is_Di:
                            temp_G.add_edge(j, i)
        self.G = copy.deepcopy(temp_G)

    # def build_graph_edge(self, a, b):
    #     device_index = 1
    #     self.G = nx.Graph()
    #
    #     # 添加节点
    #     for i in range(self.device_account):
    #         host_name = f'R{i}'
    #         device = Device(device_type=Device_type.ROUTER, host_name=host_name)
    #         self.G.add_node(i, device=device)
    #
    #     # 添加边
    #     n = self.device_account
    #     edge_probability = a / (n * (n - 1) // 2)
    #
    #     temp_G = copy.deepcopy(self.G)
    #     edge_account = 0
    #     while len(list(nx.connected_components(temp_G))) > 1:  # 避免出现孤岛,还要让边书在要求区间内
    #         temp_G = copy.deepcopy(self.G)
    #
    #         for i in range(self.device_account):
    #             for j in range(i + 1, self.device_account):  # 避免自环和重复边
    #                 if random.random() < edge_probability:  # 控制边的概率
    #                     temp_G.add_edge(i, j)
    #                     edge_account += 1
    #
    #     self.G = copy.deepcopy(temp_G)

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

    def address_distribution(self, name):
        '''
        为所有设备的物理接口和逻辑接口分配网络
        :return:
        '''
        # self.weight_matrix = np.zeros((self.device_account, self.device_account), dtype=int)
        for node1, node2, data in self.G.edges(data=True):  # 对物理接口进行赋值
            prefix, mask = self.ip_distributor.assign_address(Interface_type.GigabitEthernet)
            ip1 = str(ipaddress.IPv4Address(int(prefix) + 1))
            ip2 = str(ipaddress.IPv4Address(int(prefix) + 2))

            device1 = self.G.nodes[node1]['device']
            weight1 = self.weight_matrix[node1][node2]
            device1.add_interface(interface_type=Interface_type.GigabitEthernet, network_add=ip1, prefix=str(prefix),
                                  mask=mask, weight=weight1)

            device2 = self.G.nodes[node2]['device']
            weight2 = self.weight_matrix[node2][node1]
            device2.add_interface(interface_type=Interface_type.GigabitEthernet, network_add=ip2, prefix=str(prefix),
                                  mask=mask, weight=weight2)
            if self.is_Di:
                data['weight'] = weight1
            else:
                data['weight'] = weight1 + weight2
            # self.weight_matrix[node1][node2] = weight1  # 出端口权重
            # self.weight_matrix[node2][node1] = weight2

        for node in self.G.nodes:  # 对逻辑接口进行赋值
            prefix, mask = self.ip_distributor.assign_address(Interface_type.Loopback)
            device = self.G.nodes[node]['device']
            device.add_interface(interface_type=Interface_type.Loopback, network_add=str(prefix), prefix=str(prefix),
                                 mask=mask, weight=0)

        # np.savetxt(os.path.join(self.network_root, "weight_matrix.pkl"), self.weight_matrix, fmt='%d', delimiter='\t')
        # with open(os.path.join(self.network_root, f"{name}.pkl"), 'wb') as file:
        #     pickle.dump(self.weight_matrix, file)

    def gen_ospf_config(self):
        self.address_distribution(name='weight_matrix')
        for node in self.G.nodes:
            device = self.G.nodes[node]['device']
            device.make_ospf_config_file(self.network_root, self.need_embeded)

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

    def get_query_and_answer(self,cur_node,query_type:QueryType):
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
                    'interface':interface.get_interface_name(),
                    'address':str(interface.address),
                    'mask':str(interface.mask)
                })
            answer='\n'.join(answer_list)
            return query,answer,json.dumps(label_list)
        elif query_type == QueryType.NETWORK.value:
            this_device = self.G.nodes[cur_node]['device']
            query = f"List all the Ipv4 network of the {this_device.host_name} device"
            answer_list = []
            label_list = []
            answer_list.append(f"there are {len(this_device.networks)} networks:")
            for network in this_device.networks:
                answer_list.append(f"{network['prefix']} {network['mask']}")
                label_list.append({
                    'prefix':str(network['prefix']),
                    'mask':str(network['mask'])
                })
            answer = '\n'.join(answer_list)
            return query, answer, json.dumps(label_list)
        else:
            raise Exception("不存在的query_type!")




    def random_weight(self):
        # 随机权重矩阵,只更新0.5边
        if self.weight_matrix is None:
            self.weight_matrix = np.full((self.device_account, self.device_account), np.inf)
            for u, v, data in self.G.edges(data=True):
                weight_int = random.randint(1, self.max_weight)  # 随机权重整数
                data['weight'] = weight_int
                self.weight_matrix[u][v] = weight_int
            return self.weight_matrix

        num_edges = len(self.G.edges())
        change_edges = random.sample(list(self.G.edges(data=True)), math.ceil(num_edges * 0.5))  # 采样十分之一的边改变
        for u, v, data in change_edges:
            weight_int = random.randint(1, self.max_weight)  # 随机权重整数
            data['weight'] = weight_int
            self.weight_matrix[u][v] = weight_int
        return self.weight_matrix


if __name__ == '__main__':
    builder = NetworkBuilder(10, 0)
    builder.build_graph()
    builder.draw_graph()
    builder.gen_ospf_config()
    builder.save_pkl()
    # builder.load_pyg()
    builder.save_pkl()
