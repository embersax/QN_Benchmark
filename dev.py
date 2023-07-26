# from topo.Topo import Topo
# from algorithm.QPATH import QPath

# netTopology = Topo.generateString(5, 0.6, 5, 0.1, 6)
# topo = Topo(netTopology)

# algo = QPath(topo, 0)

# algo.P2

# path = algo.returns_shortest_path(topo.nodes[0], topo.nodes[3])
# print(path)
# print(algo.min_fidelity_link(path))

from utils.CollectionUtils import MinHeap

pq = MinHeap()
assert pq.get_length() == 0
pq.push(2, [2,3,4])
pq.push(0, [4, 3])
pq.push(1, [])
assert pq.get_length() == 3
assert pq.pop() == (0, [4, 3])
assert pq.get_length() == 2
assert pq.pop() == (1, [])
assert pq.get_length() == 1
assert pq.pop() == (2, [2,3,4])
assert pq.get_length() == 0
