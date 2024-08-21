# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/26 20:36
@Auth ： xiaolongtuan
@File ：gen_ospf_dataset.py
"""
import concurrent
import copy
import os
import pickle
from tqdm import tqdm
from ospf.NetworkBuilder import NetworkBuilder
import pandas as pd
import math

from utils.routing_service import get_routing_sim

data_sizes = [
    {
        'sub_directorie': 'm',
        'account': 1000,
        'node': 15
    },
]
max_workers = 10
column_names = ['src', 'pre_config', 'cur_config', 'dst', 'pre_path', 'cur_path', 'changed']


def add_qos_constraint(row):
    qos_constraint = "Now we need to ensure that the end-to-end delay of the traffic from node {} to node {} is less than {}ms, the average jitter is less than {}ms, and the packet loss rate is less than {:.2f}%".format(
        row['src'], row['dst'], math.ceil(row['avgDelay'] * 10) * 100, math.ceil(row['avgJitter'] * 100) * 10,
                                row['pkgLossRate'] * 100)
    return pd.Series([qos_constraint], index=['qos_constraint'])


def builder_init(node_account):
    builder = NetworkBuilder(node_account, f'ospf_update_single_topo_dataset/m/0')
    builder.build_graph()

    builder.draw_dev_graph()
    builder.draw_graph()
    return builder


def ospf_net_gen(net_id, origen_builder, sub_directorie='m'):
    builder = copy.deepcopy(origen_builder)
    builder.network_root = f'ospf_update_single_topo_dataset/m/{net_id}'
    builder.builder_init()

    df = pd.DataFrame(columns=column_names)
    pre_weight_matrix = builder.weight_matrix
    with open(os.path.join(f'ospf_update_single_topo_dataset/{sub_directorie}/{net_id}', "pre_weight_matrix.pkl"),
              'wb') as file:
        pickle.dump(pre_weight_matrix, file)
    builder.gen_ospf_config()
    old_path_record = builder.path_record()  # 旧的路径

    df_index = len(df)
    for j in builder.G.nodes():
        node = {
            'src': j,
            'pre_config': builder.G.nodes[j]['device'].config,
            'cur_config': '',
            'dst': 0,
            'pre_path': [],
            'cur_path': '',
            'changed': 0,
        }
        df.loc[df_index + j] = node

    # 更新 权重，只改了3-5个数值
    cur_weight_matrix = builder.random_weight()

    with open(os.path.join(f'ospf_update_single_topo_dataset/{sub_directorie}/{net_id}', "cur_weight_matrix.pkl"),
              'wb') as file:
        pickle.dump(cur_weight_matrix, file)
    builder.gen_ospf_config()
    builder.gen_ned_file()
    builder.gen_ini_file()

    for j, dst in builder.node_dst_map.items():
        node = df.loc[df_index + j].to_dict()

        node['dst'] = dst
        old_path = old_path_record[j][dst]
        node['pre_path'] = old_path

        cur_path = builder.get_path(j, dst)
        node['cur_path'] = cur_path

        if cur_path == old_path:
            node['changed'] = 0  # 路由需求未改变
        else:
            node['changed'] = 1

        node['cur_config'] = builder.G.nodes[j]['device'].config
        df.loc[df_index + j] = node

    ned_path = os.path.join(f'ospf_update_single_topo_dataset/omnet_file/{sub_directorie}/networks',
                            f"Myned{str(net_id)}.ned")
    ini_path = os.path.join(f'ospf_update_single_topo_dataset/omnet_file/{sub_directorie}/ini_dir',
                            f"omnetpp{str(net_id)}.ini")
    qos_dic_list = get_routing_sim(net_id=str(net_id), ned_path=ned_path, ini_path=ini_path)
    qos_df = pd.DataFrame(qos_dic_list)
    new_df = pd.merge(df, qos_df, on=['src', 'dst'])

    qos_constraint_values = new_df.apply(add_qos_constraint, axis=1)
    new_df['qos_constraint'] = qos_constraint_values

    new_df.to_json(f"ospf_update_single_topo_dataset/{sub_directorie}/{net_id}/dataset.jsonl", orient="records",
                   lines=True)
    return 1


pbar = tqdm(total=sum(entry['account'] for entry in data_sizes))
builder = builder_init(node_account=15)
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {}
    for data_size in data_sizes:
        for i in range(data_size['account']):
            sub_directorie = data_size['sub_directorie']
            node_account = data_size['node']
            futures[executor.submit(ospf_net_gen, i, builder, sub_directorie)] = i

    for future in concurrent.futures.as_completed(futures):
        # 每完成一个任务，更新进度条
        task = futures[future]
        try:
            result = future.result()
            if result == 1:
                pbar.update(1)
            else:
                print(f"Error: netid = {task}")
        except Exception as exc:
            print(exc)
            # print(f"netid {task} generated an exception: {exc}")
pbar.close()
