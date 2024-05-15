# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/5 15:37
@Auth ： xiaolongtuan
@File ：gen_dataset_embeded.py
使用GNN编码ospf数据集得到适用于GraphTranslator的编码数据集
"""
import json
import pickle

import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm

from graphSAGE.GATnet import GATNet
from graphSAGE.GCNnet import GCNNet
from graphSAGE.SAGEnet import SAGENet

# 预训练的图节点嵌入模型
model_parameters_path = 'graphSAGE/model/SAGE_graphsage.pth'  # 替换为你的模型参数文件路径

loaded_parameters = torch.load(model_parameters_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = 'SAGE'

if net == 'GCN':
    model = GCNNet(768, 1024, 768).to(device)
elif net == 'SAGE':
    model = SAGENet(768, 1024, 768).to(device)
elif net == 'GAT':
    model = GATNet(768, 1024,2, 768).to(device)
else:
    raise Exception('未选择网络类型')

model.load_state_dict(loaded_parameters)
model.eval()

type = 2
split = 'test'
def embeding_dataset(dataset_dir):
    dataset_size = 2000
    datas = []
    user_id = 0
    for i in tqdm(range(dataset_size)):
        edge_index = pickle.load(open(f'{dataset_dir}/{split}/{i}/edges.pkl', 'rb'))  # 所有
        embeded_dataset = pd.read_json(f"{dataset_dir}/{split}/{i}/dataset_type{type}.jsonl", lines=True)

        x = torch.tensor(embeded_dataset['embeded_config']).to(device)
        edge_index = torch.tensor(edge_index).to(device)
        if type == 'global':
            h = torch.mean(model(x, edge_index),dim = 0).cpu().detach().numpy()
            str_array = [str(num) for num in h]
            str_representation = ", ".join(str_array)  # 将numpy序列化
            datas.append({
                'user_id': user_id,
                'embeding': str_representation,
                'text_input': data['node_text'],
                'title': f'R{index}',
                'dest': data['dest'],
                'path': data['path']
            })
            user_id += 1
        else:
            h = model(x, edge_index).cpu().detach().numpy()
            for index, data in embeded_dataset.iterrows():  # index 从0开始s 'path', 'dest'
                node_embeding = h[index]
                str_array = [str(num) for num in node_embeding]
                str_representation = ", ".join(str_array)
                datas.append({
                    'user_id': user_id,
                    'embeding': str_representation,
                    'text_input': data['node_text'],
                    'title': f'R{index}',
                    'dest': data['dest'],
                    'path': data['path']
                })
                user_id += 1
    datas_embeddings = pd.DataFrame(datas)
    datas_embeddings.to_csv(f'data/ospf/datas_node_embeddings_{split}_type{type}.csv', index=False)
    return datas


embeding_dataset('ospf_dataset')
