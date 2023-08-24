from topo.Topo import Topo
from algorithm.QPATH import QPath
import timeit
import random
from collections import Counter
import matplotlib.pyplot as plt

def run(n):
    n1 = topo.nodes[random.randint(0, n-1)]
    n2 = topo.nodes[random.randint(0, n-1)]
    return algo.P2(n1, n2, 1)


nodes = [100, 200]
times = []
for n in nodes:
    netTopology = Topo.generateString(n, 0.6, 5, 0.1, 6)
    topo = Topo(netTopology)
    algo = QPath(topo, 0.6)
    # Read timeit docs for why take min
    times.append(min(timeit.repeat(stmt="run(n)",setup="from dev import run", globals=globals(), number=3)))
plt.bar(nodes, [time*1000 for time in times])
_, ax = plt.subplots()
ax.set_xlabel = 'ms'

plt.show()
# n = 100

# netTopology = Topo.generateString(n, 0.6, 5, 0.1, 1)
# topo = Topo(netTopology)


# algo = QPath(topo, 0.95)
# sols = algo.P2(topo.nodes[random.randint(0, n-1)], topo.nodes[random.randint(0,n-1)], 3)
# print(sols)

# for sol in sols:
#     print(f"Cost: {sol[0].cost}")
#     print(f"Path: {sol[0].path}")
#     print(f"Purification Decisions: {sol[0].pur_dec}")
#     print(f"Times Takeable: {sol[1]}")
# assert d[topo.nodes[0], topo.nodes[3]] >= sol[1]*sol[0].cost # Cap > cost

# print(algo.P2(topo.nodes[0], topo.nodes[3], 1))

