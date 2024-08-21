# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/23 22:04
@Auth ： xiaolongtuan
@File ：test.py
"""
import os
from pathlib import Path

import networkx as nx

g = nx.DiGraph()
g.add_edge(0, 1)
nx.draw_networkx_edge_labels(g, label_pos=0.8, pos=pos, edge_labels=interface_labe, ax=ax)
print(g.edges())
