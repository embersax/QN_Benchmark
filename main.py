from  topo.Topo import Topo
from math import exp
from utils.utils import *
from configs import randGenSeed
from random import shuffle
import random
from itertools import combinations
from algorithm import OnlineAlgorithm

from itertools import combinations
from algorithm.OnlineAlgorithm import OnlineAlgorithm


def simpleTest():
    netTopology = Topo.generateString(30, 0.6, 5, 0.1, 6)


    alphas = []
    p = [.8, .5]
    for expectedAvgP in p:
        alpha = step = .1
        lastAdd = True

        while True:
            lines = list(netTopology.split('\n'))
            lines[1] = str(alpha)
            topo = Topo('\n'.join(lines))
            avgP = float(sum(list(map(lambda it: exp(-alpha * length([it.n1.loc, it.n2.loc])), topo.links)))) / len(topo.links)

            if abs(avgP - expectedAvgP) / expectedAvgP < .001:  break

            elif avgP > expectedAvgP:
                if not lastAdd: step /= 2
                alpha += step
                lastAdd = True

            else:
                if lastAdd: step /= 2
                alpha -= step
                lastAdd = False

        alphas.append(alpha)

    repeat = 1000

    def funInLine135(combs, nsd, repeat):
        res = []
        for i in range(1, repeat+1):
            shuffle(combs)
            res.append(list(map(lambda it: (it[0].id, it[1].id), combs[:nsd])))
        return res

    testSetIter, testSet = [i for i in range(9, 11)], []
    for nsd in testSetIter:
        combs = list(combinations(topo.nodes, 2))
        testSet.append(funInLine135(combs, nsd, repeat))

    def helper(x, solver):
        # x  list(list([int,int]))
        results = []
        for item in x:
            #         item list([int,int])
            result = solver.work(list(map(lambda x: [topo.nodes[x[0]], topo.nodes[x[1]]], item)))
            #         result : list([int,int])
            # avgEntanglements = sum([x[0] for x in result]) / len(result)
            # avgEntangled = sum([x[1] for x in result]) / len(result)
            results.append(x)
        avgEntanglements = sum([x[0] for x in results]) / len(results)
        avgEntangled = sum([x[1] for x in results]) / len(results)
        return results

    for P, a in list(zip(p,alphas)):
        s = Topo(netTopology)
        ss = Topo(netTopology).toConfig()
        tmp_str = listtoString(list(map(lambda item: str(a) if item[0] == 1 else item[1],
                                        enumerate(Topo(netTopology).toString().splitlines()))), "\n")
        topo = Topo(tmp_str)
        # aa =  OnlineAlgorithm(Topo(topo))
        algorithms = [
            OnlineAlgorithm(Topo(topo))
        ]
        results = [[] * len(algorithms)]
        # here to test I just loop over algorithms list to test, and will implemment multithreads(parallelStream()) as Shouqian later.
        for solver in algorithms:
            topo = solver.topo
            solver.prepare()
            results[algorithms.index(solver)].extend(
                list(map(lambda x: helper(x,solver)
                    ,testSet[1:]
                ))
            )


if __name__ == "__main__":
    simpleTest()
