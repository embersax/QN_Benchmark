from topo.Topo import Topo
from algorithm.QPATH import QPath


for i in range(1):
    netTopology = Topo.generateString(5, 0.6, 5, 0.1, 6)
    topo = Topo(netTopology)

    from collections import Counter
    l = []
    for link in topo.links:
        l.append((link.n1, link.n2))
    d = Counter(l)
    print(d)


    algo = QPath(topo, 0.95)


    sols = algo.P2(topo.nodes[0], topo.nodes[3], 3)
    for sol in sols:
        print(f"Cost: {sol[0].cost}")
        print(f"Path: {sol[0].path}")
        print(f"Purification Decisions: {sol[0].pur_dec}")
        print(f"Times Takeable: {sol[1]}")
    assert d[topo.nodes[0], topo.nodes[3]] >= sol[1]*sol[0].cost # Cap > cost
# print(res)
# print(d[topo.nodes[0], topo.nodes[3]])

# print(algo.P2(topo.nodes[0], topo.nodes[3], 1))

