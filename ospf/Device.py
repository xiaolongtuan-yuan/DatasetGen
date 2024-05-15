# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/10 11:19
@Auth ： xiaolongtuan
@File ：Device.py
"""
import os.path
from enum import Enum, auto
import ipaddress

import torch
from transformers import BertTokenizer, BertModel

class Interface_type(Enum):
    Loopback = auto()
    GigabitEthernet = auto()


class Device_type(Enum):
    ROUTER = auto()
    SWITCH = auto()

need_embeded = True
if need_embeded:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "google-bert/bert-base-uncased"
    local_path = './bert-base-uncased/models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
    tokenizer = BertTokenizer.from_pretrained(local_path)
    bert_model = BertModel.from_pretrained(local_path).to(device)


def bert_encoder(config):
    inputs = tokenizer(config, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 使用BERT模型对编码后的文本进行forward传递，获取输出
    with torch.no_grad():
        outputs = bert_model(**inputs)

        # 获取最后一层的隐藏状态
        last_hidden_state = outputs.last_hidden_state

        # 获取 [CLS] 标记的隐藏状态作为整个句子的表示
        sentence_representation = last_hidden_state[:, 0, :].view(-1)
        return sentence_representation

class Config_Template:
    config_template = "interface {}{}/0\n ip address {} {}\n negotiation auto\n ip ospf cost {}"
    loop_config_template = "interface {}{}\n ip address {} {}\n negotiation auto"
    BGP_template = "router bgp {}"
    BGP_network_template = " network {} mask {}"
    BGP_neighbor_template = " neighbor {} remote-as {}"
    OSPF_process = 'router ospf 1'
    OSPF_network_template = 'network {} {} area 0'


class Interface:
    def __init__(self, interface_type, seq: int, address: str, mask: str, weight: int):
        self.interface_type = interface_type
        self.seq = seq
        self.address = address
        self.mask = mask
        self.weight = weight

    def gen_interface_config(self):
        if self.interface_type == Interface_type.GigabitEthernet:
            return Config_Template.config_template.format(self.interface_type.name, self.seq, self.address, self.mask,
                                                          self.weight)
        elif self.interface_type == Interface_type.Loopback:
            return Config_Template.loop_config_template.format(self.interface_type.name, 0, self.address,
                                                               self.mask)
        else:
            raise Exception('接口类型错误！')

    def get_interface_name(self):
        if self.interface_type == Interface_type.GigabitEthernet:
            return f"{self.interface_type.name}{self.seq}/0"
        elif self.interface_type == Interface_type.Loopback:
            return f"{self.interface_type.name}0/0"


class Device:
    def __init__(self, device_type: Device_type, network_root="", host_name=""):
        self.device_type = device_type
        self.network_root = network_root
        self.host_name = host_name
        self.interface_list = []
        self.interface_index = 0
        self.networks = []
        self.neighbors = []

    def make_ospf_config_file(self, network_root,need_embeded = False):
        self.network_root = network_root
        basic_config = []
        basic_config.append(f"hostname {self.host_name}")
        # 接口配置
        for interface in self.interface_list:
            basic_config.append(interface.gen_interface_config())
        config = "\n!\n".join(basic_config) + "\n!"

        # ospf 网络宣告
        ospf_config_list = []
        ospf_config_list.append(Config_Template.OSPF_process)
        # for network in self.networks:  # 网段宣告
        #     ospf_config_list.append(Config_Template.OSPF_network_template.format(network['prefix'], network['mask']))
        for interface in self.interface_list:  # 精准宣告
            ospf_config_list.append(Config_Template.OSPF_network_template.format(str(interface.address), '0.0.0.0'))
        OSPF_config = "\n".join(ospf_config_list)
        basic_config.append(OSPF_config)

        config = "\n!\n".join(basic_config) + "\n!"
        self.config = config
        if need_embeded:
            self.embeded_config = bert_encoder(config).cpu()
        torch.cuda.empty_cache()
        with open(os.path.join(str(self.network_root), "configs", self.host_name + ".cfg"), "w") as config_file:
            config_file.write(config)
            # print(f"创建{self.host_name}配置文件")
            return config
        return ''

    def add_interface(self, interface_type: Interface_type, network_add: str, prefix: str, mask: str, weight: int):
        interface = Interface(interface_type, self.interface_index, network_add, mask, weight)
        self.interface_index += 1
        self.interface_list.append(interface)
        self.networks.append({
            "prefix": prefix,
            "mask": mask
        })  # 记录要宣告的网络