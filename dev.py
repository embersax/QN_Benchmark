from topo.Topo import Topo
from algorithm.QLEAP import QLeap
from collections import Counter
import matplotlib.pyplot as plt
import random

n = 300

netTopology = Topo.generateString(n, 0.6, 5, 0.1, 6)
topo = Topo(netTopology)
algo = QLeap(topo, 0.6)

print(algo.P2(topo.nodes[random.randint(0, n-1)], topo.nodes[random.randint(0, n-1)], 1))
