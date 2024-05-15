# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/26 20:36
@Auth ： xiaolongtuan
@File ：gen_ospf_dataset.py
生成适用于GraphTranslator的ospf数据集
不仅限于路由推理，还包括单个设备的配置描述，对于仅配置了ospf的设备而言，只包括：设备名、接口信息、ospf配置（进程、网络）
"""
import json
import os
import pickle
import random

import numpy as np
from tqdm import tqdm
from ospf.NetworkBuilder import NetworkBuilder
import pandas as pd

type = 2
datasize = 2
split = 'train'
is_Di=False

# graph_size = [20, 30]
column_names = ['node_ids', 'config','query','query_type', 'answer','label']


def get_max_numbered_folder(directory):
    max_number = None
    for folder_name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, folder_name)):
            try:
                folder_number = int(folder_name)
                if max_number is None or folder_number > max_number:
                    max_number = folder_number
            except ValueError:
                pass
    return 0 if max_number is None else max_number + 1


START = get_max_numbered_folder('ospf_understand_dataset')

for i in tqdm(range(0, datasize)):
    df = pd.DataFrame(columns=column_names)
    node_account = 20
    edges = []
    edges.append([])
    edges.append([])
    node_text = []

    builder = NetworkBuilder(node_account, f'ospf_understand_dataset/{split}/{i}', need_embeded=False, is_Di=is_Di)
    builder.build_graph()
    builder.random_weight()
    builder.gen_ospf_config()

    probabilities = [0.6, 0.2, 0.2]
    indices = [1, 2, 3]
    for j in range(node_account):
        query_type = random.choices(indices, weights=probabilities, k=1)[0]
        query, answer,label = builder.get_query_and_answer(j,query_type)

        node = {
            'node_ids': j,
            'config': builder.G.nodes[j]['device'].config,
            'query': query,
            'query_type': query_type,
            'answer': answer,
            'label':label
        }
        df.loc[len(df)] = node
    builder.record_edges(0, edges)
    with open(f'ospf_understand_dataset/{split}/{i}/edges.pkl', 'wb') as f:
        pickle.dump(np.array(edges), f)

    df.to_json(f"ospf_understand_dataset/{split}/{i}/dataset_type{type}.jsonl", orient="records", lines=True)

dataset_info = {
    'type': split,
    'size': datasize
}
with open(f'ospf_understand_dataset/{split}/dataset_info.json', "w") as f:
    json.dump(dataset_info, f)
