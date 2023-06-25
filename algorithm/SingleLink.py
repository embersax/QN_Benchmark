# It looks like single link is an instantiation of the OnlineAlgorithm class.
from .OnlineAlgorithm import OnlineAlgorithm
from ..topo.Topo import Topo
from ..topo.Node import to,Edge


# It has to be done a little differently in Python
# Theoretically, it should work if we call any of the OnlineAlgorithm functions.
class SingleLink(OnlineAlgorithm):
    def __init__(self, topo, allowRecoveryPaths=True):
        super().__init__(topo, allowRecoveryPaths=True)
        prov = {}
        self.topo.links = []
        for link in topo.links:
            edge = Edge(link.node1, link.node2)
            if edge not in self.topo.links:
                prov[edge] = [link]
            else:
                prov[edge].append(link)
        for pair in prov:
            first = prov[pair][0]
            self.topo.links.append(first)
        self.allowRecoveryPaths = allowRecoveryPaths
        self.name = 'SL'.join('' if allowRecoveryPaths else '-R')






