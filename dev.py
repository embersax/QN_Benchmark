from topo.Topo import Topo
from algorithm.QLEAP import QLeap
import timeit
import random
from collections import Counter
import matplotlib.pyplot as plt



def run(n):
    n1 = topo.nodes[random.randint(0, n-1)]
    n2 = topo.nodes[random.randint(0, n-1)]
    return algo.P2(n1, n2, 1)


nodes = [100, 200, 300, 400, 500]
times = []
for n in nodes:
    netTopology = Topo.generateString(n, 0.6, 5, 0.1, 6)
    topo = Topo(netTopology)
    algo = QLeap(topo, 0.6)
    # Read timeit docs for why take min
    times.append(min(timeit.repeat(stmt="run(n)",setup="from dev import run", globals=globals(), number=1)))

plt.bar(nodes, [i * 1000 for i in times])
plt.xlabel('Number of Nodes')
plt.ylabel('Time in ms')
plt.show()