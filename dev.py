from topo.Topo import Topo
from algorithm.QLEAP import QLeap
from collections import Counter
import matplotlib.pyplot as plt
import random

n = 100

netTopology = Topo.generateString(n, 0.6, 5, 0.1, 6)
topo = Topo(netTopology)
algo = QLeap(topo, 0.6)

a = algo.P2(topo.nodes[random.randint(0, n-1)], topo.nodes[random.randint(0, n-1)], 1)
print(a)