# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/26 20:36
@Auth ： xiaolongtuan
@File ：gen_ospf_dataset.py
"""
import os
import pickle
import random

import numpy as np
from tqdm import tqdm
from ospf.NetworkBuilder import NetworkBuilder
import pandas as pd

data_sizes = [
    {
        'sub_directorie': 'm',
        'account': 1000,
        'node': 15
    },
]

pbar = tqdm(total=sum(entry['account'] for entry in data_sizes))
column_names = ['node_ids', 'pre_config', 'cur_config', 'dst', 'pre_path', 'cur_path', 'changed']


split = []
edge_split = []
node_text = []
for data_size in data_sizes:
    for i in range(data_size['account']):
        df = pd.DataFrame(columns=column_names)

        # node_account = random.randint(*graph_size)
        node_account = data_size['node']
        sub_directorie = data_size['sub_directorie']

        builder = NetworkBuilder(node_account, f'ospf_update_dataset/{sub_directorie}/{i}')
        builder.build_graph()
        # builder.draw_graph()
        pre_weight_matrix = builder.weight_matrix
        with open(os.path.join(f'ospf_update_dataset/{sub_directorie}/{i}', "pre_weight_matrix.pkl"), 'wb') as file:
            pickle.dump(pre_weight_matrix, file)
        builder.gen_ospf_config()
        old_path_record = builder.path_record()  # 旧的路径

        df_index = len(df)
        for j in builder.G.nodes():
            node = {
                'node_ids': j,
                'pre_config': builder.G.nodes[j]['device'].config,
                'cur_config': '',
                'dst': 0,
                'pre_path': [],
                'cur_path': '',
                'changed': 0,
                # 'delay':,
                # 'jitter':,
                # 'pkg_loss_rate':
            }
            df.loc[df_index + j] = node

        # 更新 权重，只改了3-5个数值
        cur_weight_matrix = builder.random_weight()

        with open(os.path.join(f'ospf_update_dataset/{sub_directorie}/{i}', "cur_weight_matrix.pkl"), 'wb') as file:
            pickle.dump(cur_weight_matrix, file)
        builder.gen_ospf_config()
        builder.gen_ned_file()
        builder.gen_ini_file()
        # builder.draw_graph()
        # builder.draw_dev_graph()
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
        pbar.update(1)
        df.to_json(f"ospf_update_dataset/{sub_directorie}/{i}/dataset.jsonl", orient="records", lines=True)

pbar.close()
