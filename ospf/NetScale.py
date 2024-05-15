# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/9 20:04
@Auth ： xiaolongtuan
@File ：NetScale.py
"""


class PodScale:
    def __init__(self, name, Rsw_Count, Fsw_Count):
        self.name = name
        self.Rsw_Count = Rsw_Count
        self.Fsw_Count = Fsw_Count


class SpScale:
    def __init__(self, name, Ssw_count):
        self.name = name
        self.Ssw_count = Ssw_count

class NetScale:
    def __init__(self, dic):
        net_name = dic.get('net_name')
        Pod_Count = dic.get('Pod_Count')
        Fsw_Count = dic.get('Fsw_Count')
        self.describe = []
        self.describe.append(f'''网络包含{Pod_Count}个service pod和{Fsw_Count}个spine plane''')
        pods = []
        self.device_list = []
        for pod in dic.get('pods'):
            pod_scale = PodScale(pod.get('name'), pod.get('Rsw_Count'), Fsw_Count)
            pod_device_list = []
            for i in range(pod_scale.Fsw_Count):
                pod_device_list.append(pod_scale.name + f"fsw{i}")
                self.device_list.append(pod_scale.name + f"fsw{i}")
            for i in range(pod_scale.Rsw_Count):
                pod_device_list.append(pod_scale.name + f"rsw{i}")
                self.device_list.append(pod_scale.name + f"rsw{i}")
            self.describe.append(f"{pod_scale.name}包含设备：{','.join(pod_device_list)}")
            pods.append(pod_scale)
        SPs = []
        for sp in dic.get('SPs'):
            sp_scale = SpScale(sp.get('name'), sp.get('Ssw_count'))
            plane_device_list = []
            for i in range(sp_scale.Ssw_count):
                plane_device_list.append(sp_scale.name + f"ssw{i}")
                self.device_list.append(sp_scale.name + f"ssw{i}")
            self.describe.append(f"{sp_scale.name}包含设备：{','.join(plane_device_list)}")
            SPs.append(sp_scale)
        if not Fsw_Count == len(SPs):
            raise Exception("FSW与Spain plane数量不一致")
        self.build(net_name, Pod_Count, Fsw_Count, pods, SPs)

    def build(self, net_name, Pod_Count, Fsw_Count, pods, SPs):
        self.net_name = net_name
        self.Pod_Count = Pod_Count
        self.Fsw_Count = Fsw_Count
        self.pods = pods
        self.SPs = SPs

    def net_describe(self):
        return "\n".join(self.describe)

if __name__ == '__main__':
    print(1)
