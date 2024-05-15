# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/5 16:08
@Auth ： xiaolongtuan
@File ：SAGEnet.py
"""
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GATNet(nn.Module):
    def __init__(self, in_dim, hid_dim, head, out_dim):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_dim, hid_dim, head)
        self.conv2 = GATConv(hid_dim*head, out_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x