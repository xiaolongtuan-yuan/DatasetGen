# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/5 16:08
@Auth ： xiaolongtuan
@File ：GCNnet.py
"""
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GCNConv
import torch.nn.functional as F

class GCNNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x