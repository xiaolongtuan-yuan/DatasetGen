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


def create_dataset_directory(root_dir, tree_structure):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for directory, subdirectories in tree_structure.items():
        directory_path = os.path.join(root_dir, directory)
        if subdirectories is None:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
        else:
            create_dataset_directory(directory_path, subdirectories)


tree_structure = {
    "dataset(4-4)": {
        "m": None,
        "l": None
    }
}

# 创建dataset目录：dataset(4-4)
create_dataset_directory('./', tree_structure)
# 3种网络规模
data_sizes = [
    {
        'sub_directorie': 'm',
        'account': 1000,
        'node': 15,
        'edge': (20, 25)
    },
    {
        'sub_directorie': 'l',
        'account': 1000,
        'node': 20,
        'edge': (25, 30)
    }
]
total_account = sum(data['account'] for data in data_sizes)

pbar = tqdm(total=total_account)

split = []
edge_split = []
node_text = []
for data_size in data_sizes:
    for i in range(data_size['account']):
        # node_account = random.randint(*graph_size)
        node_account = data_size['node']
        sub_directorie = data_size['sub_directorie']

        builder = NetworkBuilder(node_account, f'dataset(4-4)/{sub_directorie}/{i}')
        builder.build_graph()
        pre_weight_matrix = builder.random_weight()
        with open(os.path.join(f'dataset(4-4)/{sub_directorie}/{i}', "pre_weight_matrix.pkl"), 'wb') as file:
            pickle.dump(pre_weight_matrix, file)
        builder.gen_ospf_config()
        pbar.update(1)

pbar.close()
