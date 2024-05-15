# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/19 10:47
@Auth ： xiaolongtuan
@File ：ryu_manager.py
"""

from ryu.base import app_manager
from ryu.ofproto import ofproto_v1_3
from webob import Response
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet, ipv4
from ryu.lib import dpid as dpid_lib

from network_model import NetworkModel

# 导入Ryu框架中的基础模块，这些模块提供了创建和管理Ryu应用的基础功能。
# ofp_event, CONFIG_DISPATCHER, MAIN_DISPATCHER 和 ofproto_v1_3 是用于处理OpenFlow事件和协议的模块。
# packet 是用于创建和解析网络包的模块。

url = '/simpleswitch/random_weight/{index}'


class SimpleSwitchController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super(SimpleSwitchController, self).__init__(req, link, data, **config)
        self.app = data['app']

    @route('simpleswitch', url, methods=['GET'])
    def ramdom_weight_of_index(self, req, **kwargs):
        print(f"gen dataset of {kwargs['index']}")
        dataset_index = kwargs['index']
        self.app.dataset_index = dataset_index
        self.app.net_model.random_weight(dataset_index)
        return Response(content_type='application/json')


class SimpleSwitch(app_manager.RyuApp):  # 定义了一个名为SimpleSwitch的类，它继承自RyuApp，这是Ryu应用的基础类。
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]  # OFP_VERSIONS 属性指定了这个应用支持的OpenFlow版本。
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch, self).__init__(*args, **kwargs)

        self.net_model = NetworkModel()
        self.net_model.load_net_graph("datas/topology.graphml")
        self.dataset_index = 0
        wsgi = kwargs['wsgi']
        wsgi.register(SimpleSwitchController, {'app': self})

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)  # 定义了一个事件处理器，当交换机发送SwitchFeatures消息时被调用。
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath  # ev.msg.datapath 获取交换机的数据路径对象。
        ofproto = datapath.ofproto  # ofproto 获取当前数据路径的OpenFlow协议版本。
        parser = datapath.ofproto_parser  # parser 获取用于解析OpenFlow消息的解析器。

        # Install table-miss flow entry
        match = parser.OFPMatch()  # 创建一个匹配所有流量的流表项，并将其动作设置为向控制器发送Packet-In消息。
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]  # OFPP_CONTROLLER 表示输出到控制器。OFPCML_NO_BUFFER 表示不使用缓冲区，直接发送完整包数据。
        self.add_flow(datapath, 0, match, actions)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)  # 定义了一个事件处理器，当交换机接收到Packet-In消息时被调用。
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']  # msg.match['in_port'] 获取Packet-In消息的输入端口。

        # 解析Packet-In消息中的数据包，获取以太网帧和源MAC地址。
        pkt = packet.Packet(msg.data)
        v4 = pkt.get_protocols(ipv4.ipv4)

        if v4 is not None:
            dst_ip = v4.dst
            src_ip = v4.src

            out_port = self.net_model.forward(datapath.id, v4.src, v4.dst)

            actions = [parser.OFPActionOutput(out_port)]
            match = parser.OFPMatch(in_port=in_port, eth_src=v4.src, eth_dst=v4.dst)
            self.add_flow(datapath, 1, match, actions)

        data = None if msg.buffer_id == ofproto.OFP_NO_BUFFER else msg.data
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
