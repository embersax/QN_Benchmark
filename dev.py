from topo.Topo import Topo
from algorithm.QPATH import QPath

netTopology = Topo.generateString(5, 0.6, 5, 0.1, 6)
topo = Topo(netTopology)

algo = QPath(topo, 0)

# Check to make sure the # of links == # of possible fidelities
t = 0
for link, fid_list in algo.purification_table.items():
    t += len(fid_list)
assert t == len(topo.links)

algor = QPath(topo, 2)
assert len(algor.purification_table) == 0

