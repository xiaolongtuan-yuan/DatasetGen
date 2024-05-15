# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/18 21:08
@Auth ： xiaolongtuan
@File ：gen_topo_py.py
"""

# s1 = self.addSwitch('s1',dpid='1')
node_temp = "        s{} = self.addSwitch('s{}', dpid='{}')"
link_temp = "        self.addLink(s{}, s{}, {}, {})"


class GenTopoPy():
    def __init__(self, path="topi_py_temp.txt"):
        self.node_str = []
        self.link_str = []
        with open(path, 'r') as topo_file:
            self.topo_file_temp = topo_file.read()

    def add_node(self, dpid):
        self.node_str.append(node_temp.format(dpid, dpid, dpid))

    def add_link(self, src_dpid, from_port, dis_dpid, to_port):
        self.link_str.append(link_temp.format(src_dpid, dis_dpid, from_port, to_port))

    def gen_file(self, write_path):
        node_body = "\n".join(self.node_str)
        link_body = "\n".join(self.link_str)

        file_body = self.topo_file_temp.format(node_body, link_body)
        with open(write_path, 'w') as write_file:
            write_file.write(file_body)
