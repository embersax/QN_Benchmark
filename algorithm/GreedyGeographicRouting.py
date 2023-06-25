from QN_routing.algorithm.AlgorithmBase import Algorithm
from QN_routing.topo.Topo import Topo, Edge, Path
from itertools import cycle
from QN_routing.utils.utils import length

class GreedyGeographicRouting(metaclass=Algorithm):
    def __init__(self, topo):
        self.topo = topo
        # overrides 'name' from Algorithm class
        self.name = 'Greedy_G'
        self.pathsSortedDynamically = []


    def prepare(self):
        assert self.topo.isClean()

    def P2(self):
        # needs to be debugged after GreedyHopRouting
        flag_box = [True for i in range(len(self.srcDstPairs))]
        while any(flag_box):
            count = 0
            for src, dst in cycle(self.srcDstPairs):
                tag = count % len(self.srcDstPairs)
                p = [src]
                while (True):
                    last = p[-1]
                    if (last == dst):
                        break
                    next = sorted(list(filter(lambda neighbor: (
                                                                           neighbor.remainingQubits > 2 or neighbor == dst and neighbor.remainingQubits > 1) and
                                                               len(list(filter(
                                                                   lambda x: (not x.assigned) and x.contains(neighbor),
                                                                   last.links))) > 0,
                                              last.neighbors)),key= lambda it: length(it.loc - dst.loc))[0]
                    if (next == None) or next in p:
                        break
                    p += next
                if p[-1] != dst:
                    flag_box[tag] = False
                    count += 1
                    continue
                width = self.topo.widthPhase2(p)
                if width == 0:
                    flag_box[tag] = False
                    count += 1
                    continue
                self.pathsSortedDynamically.append([0, width, p])

                for i in range(1, width + 1):
                    for n1, n2 in Path.edges(p):
                        next(x for x in n1.links if x.contains(n2) and not x.assigned).assignQubits()
                count += 1

    def P4(self):
        oldNumOfPairs = 0
        for _, width, p in self.pathsSortedDynamically:
            oldNumOfPairs = len(self.topo.getEstablishedEntanglements(p[0], p[-1]))
            tmp = list(zip(list(zip(p[0:-2], p[1:-1])), p[2:]))
            for n12, next in tmp:
                prev, n = n12
                prevLinks = list(
                    filter(lambda x: x.entangled and ((x.n1 == prev) and not x.s2 or (x.n2 == prev and not x.s1)),
                           n.links))[0:width]
                nextLinks = list(
                    filter(lambda x: x.entangled and ((x.n1 == next) and not x.s2 or (x.n2 == next and not x.s1)),
                           n.links))[0:width]

                for l1, l2 in list(zip(prevLinks, nextLinks)):
                    n.attemptSwapping(l1, l2)

        succ = len(self.topo.getEstablishedEntanglements(p[0].p[-1])) - oldNumOfPairs

        self.logWriter.write(f"{list(map(lambda x: x.id, p))} " + f"{width} {succ} ")


