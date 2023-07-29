from topo.Topo import Topo
from algorithm.QPATH import QPath

netTopology = Topo.generateString(5, 0.6, 5, 0.1, 6)
topo = Topo(netTopology)

algo = QPath(topo, 0.95)

algo.P2

print(algo.P2(topo.nodes[0], topo.nodes[3], 1))

