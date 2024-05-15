# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/19 20:14
@Auth ： xiaolongtuan
@File ：main.py
"""
import os

from tqdm import tqdm

from bgp_builder import BgpBuilder
from bgp_semantics import BgpSemantics
import numpy as np


def choose_random(arr, s=None):
    s = np.random if s is None else s
    idx = s.randint(0, len(arr))
    return arr[idx]


total_account = 10

if __name__ == '__main__':
    seed = os.getpid()
    pbar = tqdm(total=total_account)

    s = np.random.RandomState(seed=seed)
    real_world_topology = False
    num_networks = choose_random(list(range(2, 4)), s)  # 外网个数
    # num_networks = 1
    num_gateway_nodes = 2  # 每个外网连接网关个数
    num_nodes = choose_random(range(4, 10), s)  # ospf域内节点个数
    # num_nodes = 3
    sample_config_overrides = {
        "fwd": {
            "n": choose_random([2, 4, 6], s)  # 每个外网抽样的fwd个数
        },
        "reachable": {
            "n": choose_random([2, 3, 4, 5], s)  # 总个数
        },
        "trafficIsolation": {
            "n": choose_random(list(range(3, 9)), s)  # 总个数
        }
    }
    for i in range(total_account):
        builder = BgpBuilder(net_seq=i,network_root=f'data/{i}')
        builder.build_graph(num_nodes, real_world_topology, num_networks, sample_config_overrides, seed,
                            num_gateway_nodes,max_interface=8)
        # builder.draw_grap()
        # 使用graph_model建立配置文件
        builder.gen_config()
        builder.gen_ned_file()
        builder.gen_ini_file()

        pbar.update(1)
    print('finished')
    # 使用fact构造转发需求
