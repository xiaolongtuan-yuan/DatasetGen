import math
import pickle
import time

import datasets
import pandas as pd
from datasets import Dataset
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling
from transformers import BertTokenizer, BertModel

# 加载BERT模型和tokenizer
model_name = 'bert-base-uncased'  # 使用小写的uncased模型
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

def bert_encoder(sample):
    sample['embeded_config'] = tokenizer(sample['config'], add_special_tokens=False,max_length=768)
    return sample

def load_dataset(dataset_dir,i):
    with open(f'{dataset_dir}/node_text.pkl', 'rb') as f:
        node_text = pickle.load(f)
    edge_index = pickle.load(open(f'{dataset_dir}/edges.pkl', 'rb'))  # 所有
    split = pickle.load(open(f'{dataset_dir}/split.pkl', 'rb'))  # 划分每个小图的边界
    edge_split = pickle.load(open(f'{dataset_dir}/edge_split.pkl', 'rb'))  # 划分每个小图的边界
    dataset = Dataset.from_pandas(pd.read_json(f"{dataset_dir}/dataset.jsonl", lines=True))
    embeded_dataset = dataset.map(bert_encoder)
    print(embeded_dataset[0])

    data = Data(x=embeded_dataset, edge_index=torch.tensor((edge_index)),edge_label_index = torch.tensor((edge_index)), edge_label=torch.ones(len(edge_index[1])))

    edge_label_index = torch.empty(2, 0)
    edge_label = torch.empty(0)
    for i in range(len(edge_split)-1):
        begin_index = edge_split[i]
        end_index = edge_split[i+1]
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index[:,begin_index:end_index], num_nodes=20,
            num_neg_samples=(end_index - begin_index), method='sparse')
        edge_label_index = torch.cat(
            [edge_label_index,data.edge_index[:,begin_index:end_index], neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            edge_label,
            data.edge_label[begin_index:end_index],
            data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

    data.edge_label_index = edge_label_index
    data.edge_label = edge_label
    return data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = load_dataset('../ospf_sp')
data = data.to(device)

class Net(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


model = Net(768, 1024, 768).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()

    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        h = model(batch.x, batch.edge_index)
        h_src = h[batch.edge_label_index[0]]
        h_dst = h[batch.edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)

    return total_loss / data.num_nodes


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.act(out)
        out = self.linear2(out)
        return out


def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


def test():
    with torch.no_grad():
        model.eval()
        out = model(data.x, data.edge_index)

    for epoch in range(1, 501):
        LR_model.train()
        optimizer.zero_grad()
        pred = LR_model(out[data.train_idx])

        label = F.one_hot(data.y[data.train_idx], 40).float()

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

    LR_model.eval()
    val_outputs = LR_model(out[data.valid_idx])
    val_acc = compute_accuracy(val_outputs, data.y[data.valid_idx])

    test_outputs = LR_model(out[data.test_idx])
    test_acc = compute_accuracy(test_outputs, data.y[data.test_idx])

    return val_acc, test_acc


times = []
best_acc = 0
for epoch in range(10):
    start = time.time()
    input_dim = 768
    output_dim = torch.max(data.y) + 1
    LR_model = LogisticRegression(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(LR_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    loss = train()
    print("loss:", loss)

out = model(data.x, data.edge_index)
torch.save(out, "../../data/arxiv/graphsage_node_embeddings.pt")
