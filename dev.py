from topo.Topo import Topo
from algorithm.QLEAP import QLeap
from collections import Counter
import matplotlib.pyplot as plt

# netTopology = Topo.generateString(5, 0.6, 5, 0.1, 6)
# topo = Topo(netTopology)
# algo = QLeap(topo, 0.6)

# print(algo.P2(topo.nodes[1], topo.nodes[0], 1))

class Test():
    def __init__(self, n1, n2):
        self.n1 = min(n1, n2)
        self.n2 = max(n1, n2)
    def __eq__(self, other):
        if self.n1 == other.n1 and self.n2 == other.n2:
            return True
        return False
    def __hash__(self):
        return hash((self.n1, self.n2))
l = [Test(1, 2), Test(1, 2)]
s = set(l)
print(s)