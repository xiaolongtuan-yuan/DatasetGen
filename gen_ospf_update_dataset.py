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
        'sub_directorie': 's',
        'account': 5000,
        'node': 10
    },
    {
        'sub_directorie': 'm',
        'account': 5000,
        'node': 15
    },
    {
        'sub_directorie': 'l',
        'account': 5000,
        'node': 20
    }
]

pbar = tqdm(total = sum(entry['account'] for entry in data_sizes))
column_names = ['node_ids', 'pre_config', 'cur_config', 'dst', 'pre_path', 'cur_path', 'changed']
df = pd.DataFrame(columns=column_names)

edges = []
edges.append([])
edges.append([])
split = []
edge_split = []
node_text = []
for data_size in data_sizes:
    for i in range(data_size['account']):
        edges = []
        edges.append([])
        edges.append([])
        # node_account = random.randint(*graph_size)
        node_account = data_size['node']
        sub_directorie = data_size['sub_directorie']

        builder = NetworkBuilder(node_account, f'ospf_update_dataset/{sub_directorie}/{i}')
        builder.build_graph()
        # builder.draw_graph()
        pre_weight_matrix = builder.random_weight()
        with open(os.path.join(f'ospf_update_dataset/{sub_directorie}/{i}', "pre_weight_matrix.pkl"), 'wb') as file:
            pickle.dump(pre_weight_matrix, file)
        builder.gen_ospf_config()

        df_index = len(df)
        for j in range(node_account):
            dst = j
            while dst == j:
                dst = random.sample(list(builder.G.nodes), 1)[0]
            pre_path = builder.get_path(j, dst)
            node = {
                'node_ids': j,
                'pre_config': builder.G.nodes[j]['device'].config,
                'cur_config': '',
                'dst': dst,
                'pre_path': pre_path,
                'cur_path': '',
                'changed': 0
            }
            df.loc[df_index + j] = node

        cur_weight_matrix = builder.random_weight()
        with open(os.path.join(f'ospf_update_dataset/{sub_directorie}/{i}', "cur_weight_matrix.pkl"), 'wb') as file:
            pickle.dump(cur_weight_matrix, file)
        builder.gen_ospf_config()
        for j in range(node_account):
            node = df.loc[df_index + j].to_dict()

            dst = df.loc[df_index + j, 'dst']

            cur_path = builder.get_path(j, dst)
            if cur_path == node['pre_path']:
                node['changed'] = 0  # 路由需求未改变
            else:
                node['changed'] = 1

            node['cur_config'] = builder.G.nodes[j]['device'].config
            node['cur_path'] = cur_path
            df.loc[df_index + j] = node
        pbar.update(1)
        df.to_json(f"ospf_update_dataset/{sub_directorie}/{i}/dataset.jsonl", orient="records", lines=True)

pbar.close()
