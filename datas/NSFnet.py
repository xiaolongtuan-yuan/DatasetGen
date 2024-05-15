from mininet.topo import Topo

class MyTopo( Topo ):

    def build(self):

        # add hosts and switches
        s1 = self.addSwitch('s1', dpid='1')
        s2 = self.addSwitch('s2', dpid='2')
        s3 = self.addSwitch('s3', dpid='3')
        s4 = self.addSwitch('s4', dpid='4')
        s5 = self.addSwitch('s5', dpid='5')
        s6 = self.addSwitch('s6', dpid='6')
        s7 = self.addSwitch('s7', dpid='7')
        s8 = self.addSwitch('s8', dpid='8')
        s9 = self.addSwitch('s9', dpid='9')
        s10 = self.addSwitch('s10', dpid='10')
        s11 = self.addSwitch('s11', dpid='11')
        s12 = self.addSwitch('s12', dpid='12')
        s13 = self.addSwitch('s13', dpid='13')
        s14 = self.addSwitch('s14', dpid='14')
        # add links
        self.addLink(s1, s3, 0, 0)
        self.addLink(s1, s2, 1, 0)
        self.addLink(s1, s6, 2, 0)
        self.addLink(s2, s3, 1, 1)
        self.addLink(s2, s4, 2, 0)
        self.addLink(s3, s9, 2, 0)
        self.addLink(s4, s7, 1, 0)
        self.addLink(s4, s14, 2, 0)
        self.addLink(s4, s5, 3, 0)
        self.addLink(s6, s11, 1, 0)
        self.addLink(s6, s7, 2, 1)
        self.addLink(s7, s8, 2, 0)
        self.addLink(s8, s9, 1, 1)
        self.addLink(s9, s10, 2, 0)
        self.addLink(s5, s10, 1, 1)
        self.addLink(s10, s12, 2, 0)
        self.addLink(s10, s13, 3, 0)
        self.addLink(s11, s12, 1, 1)
        self.addLink(s11, s13, 2, 1)
        self.addLink(s12, s14, 2, 1)
        self.addLink(s13, s14, 2, 2)

topos = { 'mytopo': ( lambda: MyTopo() ) }