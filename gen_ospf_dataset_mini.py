# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/26 20:36
@Auth ： xiaolongtuan
@File ：gen_ospf_dataset.py
生成适用于GraphTranslator的ospf数据集
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
datasize = 2000
split = 'train'
is_Di=False

# graph_size = [20, 30]
column_names = ['node_ids', 'config', 'embeded_config', 'path', 'dest', 'node_text']


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


START = get_max_numbered_folder('ospf_dataset')

for i in tqdm(range(0, datasize)):
    df = pd.DataFrame(columns=column_names)
    node_account = 20
    edges = []
    edges.append([])
    edges.append([])
    node_text = []

    builder = NetworkBuilder(node_account, f'ospf_dataset/{split}/{i}', need_embeded=True, is_Di=is_Di)
    builder.build_graph()
    builder.random_weight()
    builder.gen_ospf_config()
    for j in range(node_account):
        dest, path = builder.get_path_from_start(j)
        node_text = builder.get_node_presentation_type2(j) if type == 2 else builder.get_node_presentation(j)  # 两种不同的文本表示方式
        node = {
            'node_ids': j,
            'config': builder.G.nodes[j]['device'].config,
            'path': path,
            'dest': dest,
            'embeded_config': builder.G.nodes[j]['device'].embeded_config.detach().numpy(),
            'node_text': node_text
        }
        df.loc[len(df)] = node
    builder.record_edges(0, edges)
    with open(f'ospf_dataset/{split}/{i}/edges.pkl', 'wb') as f:
        pickle.dump(np.array(edges), f)

    df.to_json(f"ospf_dataset/{split}/{i}/dataset_type{type}.jsonl", orient="records", lines=True)

dataset_info = {
    'type': split,
    'size': datasize
}
with open(f'ospf_dataset/{split}/dataset_info.json', "w") as f:
    json.dump(dataset_info, f)
