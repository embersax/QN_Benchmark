from topo.Topo import Topo
from algorithm.QPATH import QPath

netTopology = Topo.generateString(5, 0.6, 5, 0.1, 6)
topo = Topo(netTopology)

algo = QPath(topo, 0)

print(topo.nodes[0], topo.nodes[1])

print(algo.returns_shortest_path(topo.nodes[0], topo.nodes[3]))

