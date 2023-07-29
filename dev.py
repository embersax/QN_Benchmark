from topo.Topo import Topo
from algorithm.QPATH import QPath

netTopology = Topo.generateString(5, 0.6, 5, 0.1, 6)
topo = Topo(netTopology)

algo = QPath(topo, 0.95)

from collections import Counter
l = []
for link in topo.links:
    l.append((link.n1, link.n2))
d = Counter(l)

res =  algo.P2(topo.nodes[0], topo.nodes[3], 1) 
print(res)
print(d[topo.nodes[0], topo.nodes[3]])

