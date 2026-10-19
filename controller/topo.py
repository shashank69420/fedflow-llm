from mininet.topo import Topo

class DualPathTopo(Topo):
    def build(self):
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')
        s4 = self.addSwitch('s4')

        self.addLink(h1, s1)
        self.addLink(h2, s2)
        self.addLink(s1, s3, bw=10)
        self.addLink(s3, s2, bw=10)
        self.addLink(s1, s4, bw=50)
        self.addLink(s4, s2, bw=50)

topos = {'dualpath': (lambda: DualPathTopo())}
