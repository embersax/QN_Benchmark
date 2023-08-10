from topo.Topo import Topo
from algorithm.QPATH import QPath

netTopology = Topo.generateString(5, 0.6, 5, 0.1, 6)
topo = Topo(netTopology)

from collections import Counter
l = []
for link in topo.links:
    l.append((link.n1, link.n2))
d = Counter(l)
print(d)


algo = QPath(topo, 0.95)


print(algo.P2(topo.nodes[0], topo.nodes[3], 2))
# print(res)
# print(d[topo.nodes[0], topo.nodes[3]])

# print(algo.P2(topo.nodes[0], topo.nodes[3], 1))

