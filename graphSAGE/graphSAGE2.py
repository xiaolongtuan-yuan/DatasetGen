import json
import pickle
import random

import pandas as pd
from datasets import Dataset
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F

from torch_geometric.utils import negative_sampling
from tqdm import tqdm

from GATnet import GATNet
from GCNnet import GCNNet
from SAGEnet import SAGENet

net = 'SAGE'
type = 2
def load_dataset(dataset_dir):
    dataset_size = json.load(open(f'{dataset_dir}/dataset_info.json', 'r'))['size']
    datas = []
    for i in range(dataset_size):
        edge_index = pickle.load(open(f'{dataset_dir}/{i}/edges.pkl', 'rb'))  # 所有
        embeded_dataset = Dataset.from_pandas(pd.read_json(f"{dataset_dir}/{i}/dataset_type{type}.jsonl", lines=True))

        x = torch.tensor(embeded_dataset['embeded_config'])

        data = Data(x=x, edge_index=torch.tensor((edge_index), dtype=torch.int64),
                    edge_label_index=torch.tensor((edge_index), dtype=torch.int64),
                    edge_label=torch.ones(len(edge_index[1])))

        edge_label_index = torch.empty(2, 0, dtype=torch.int64)
        edge_label = torch.empty(0)

        neg_edge_index = negative_sampling(
            edge_index=data.edge_index, num_nodes=20,
            num_neg_samples=data.edge_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [edge_label_index, data.edge_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            edge_label,
            data.edge_label,
            data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)
        data.edge_label_index = edge_label_index
        data.edge_label = edge_label
        datas.append(data)
    return datas


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_list = load_dataset('../ospf_dataset/train')
data_size = len(data_list)
# 随机打乱数据
random.shuffle(data_list)
train_size = int(0.8 * data_size)
test_size = data_size - train_size
train_data = data_list[:train_size]
test_data = data_list[train_size:]

batch_size = 128
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)



learning_rate = 0.001
num_epochs = 500

if net == 'GCN':
    model = GCNNet(768, 1024, 768).to(device)
elif net == 'SAGE':
    model = SAGENet(768, 1024, 768).to(device)
elif net == 'GAT':
    model = GATNet(768, 1024,2, 768).to(device)
else:
    raise Exception('未选择网络类型')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = F.binary_cross_entropy_with_logits

train_loss = []
test_loss = []
test_accuracy = []


def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            h = model(batch.x, batch.edge_index)
            h_src = h[batch.edge_label_index.long()[0]]
            h_dst = h[batch.edge_label_index.long()[1]]
            pred = (h_src * h_dst).sum(dim=-1)
            loss = criterion(pred, batch.edge_label)
            total_loss += float(loss) * pred.size(0)
            # 计算准确率
            test_auc = roc_auc_score(batch.edge_label.cpu().numpy(), pred.cpu().numpy())
            correct += test_auc

    avg_loss = total_loss / len(test_loader)
    test_loss.append(avg_loss)
    accuracy = correct / len(test_loader)
    test_accuracy.append(accuracy)

    return accuracy


def train(model, train_loader, optimizer, criterion):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for batch_id, batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            h = model(batch.x, batch.edge_index)
            h_src = h[batch.edge_label_index.long()[0]]
            h_dst = h[batch.edge_label_index.long()[1]]
            pred = (h_src * h_dst).sum(dim=-1)
            loss = criterion(pred, batch.edge_label)
            progress_bar.set_postfix({'loss': loss.item()}, refresh=True)

            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        train_loss.append(avg_loss)

        accuracy = test(model, test_loader, criterion)
        progress_bar.set_postfix({'avg_loss': avg_loss, 'acc': accuracy}, refresh=True)
        print(f'avg_loss: {avg_loss}, acc: {accuracy}')

    with open(f'res/{net}_train_loss.pkl', 'wb') as f:
        pickle.dump(train_loss, f)
    with open(f'res/{net}_test_loss.pkl', 'wb') as f:
        pickle.dump(test_loss, f)
    with open(f'res/{net}_test_accuracy.pkl', 'wb') as f:
        pickle.dump(test_accuracy, f)


train(model, train_loader, optimizer, criterion)
torch.save(model.state_dict(), f"model/{net}_graphsage.pth")
