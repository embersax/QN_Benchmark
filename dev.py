from topo.Topo import Topo
from algorithm.QPATH import QPath
import timeit
import random
from collections import Counter

netTopology = Topo.generateString(100, 0.6, 5, 0.1, 6)
topo = Topo(netTopology)
algo = QPath(topo, 0.6)
def run():
    n1 = topo.nodes[random.randint(0, 99)]
    n2 = topo.nodes[random.randint(0, 99)]
    print(n1.id, n2.id)
    return algo.P2(n1, n2, 1)


time = timeit.timeit(stmt="run()",setup="from dev import run", number=1)
print(time)

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

