from QN_routing.algorithm.AlgorithmBase import Algorithm
from QN_routing.topo.Topo import *
from QN_routing.utils.utils import *
from QN_routing.utils.Disjoinset import *
from QN_routing.utils.CollectionUtils import PriorityQueue
import heapq
import copy
from QN_routing.topo.Node import to


# returns the priority based on the type and the element
# trying to use this as a replacement for the dynamic comparator
def getPriority(element, type):
    if type is None or type in {'SumDist', 'CR'}:
        return element
    elif type == 'MultiMetric':
        return math.log(element[0], 2) - element[1] + 10000 if element[1] == 1 else math.log(element[0], 2) - element[
            1] + 0
    elif type == 'BotCap':
        return element[1] * 10000 + element[0]

    else:
        raise RuntimeError('Invalid priority type')


class OfflinePathBasedAlgorithm(Algorithm):

    def __init__(self, topo, allowRecoveryPaths=True):
        super().__init__(topo)
        self.priorityType = None
        self.pathsSortedDynamically = []
        self.extraPaths = []
        self.fStateMetric = ReducibleLazyEvaluation(
            initilizer=lambda x: length(list_minus(x.n1.loc, x.n2.loc)) + topo.internalLength)

        def initlizar(src, dst):
            #  q is priority que and default comparator is  the value of l1 (l1, _)   ,
            #  and each element of q should be  (Double, Path)
            q = []
            # tmp1 {Edge : list(link ) }
            tmp1 = groupby_dict(lambda x: to(x.n1, x.n1), topo.links)
            #  tmp2 List(edge)
            tmp2 = list(map(lambda x: to(tmp1[x].n1, tmp1[x].n2), tmp1))
            edges = set(tmp2)
            visited = {}
            p = topo.shortestPath(edges, src, dst, self.fStateMetric)
            visited[edges] = p
            if p[1] is not None:
                if p not in q:
                    heapq.heappushpop(q, (p[0], p))

            def work():
                tmp_list = list(visited)
                for edge, p in tmp_list:
                    if p[1] is None:
                        return
                    relatedEdges = set(p[1].edges())
                    for edge in relatedEdges:
                        edges.remove(edge)
                        if edge not in visited:
                            p = topo.shortestPath(edges, src, dst, self.fStateMetric)
                            visited[edges] = p
                            if p[1] is not None and p not in q:
                                heapq.heappushpop(q, (p[0], p))
                        edges.add(edge)

            tries = 0
            while (len(q) < 50 and (tries < 100)):
                work()
                tries += 1
            tmp = []
            while (q is not []):
                p = q.pop(0)
                width = topo.widthPhase2(p[1])
                tmp.append([p[0], width, p[1]])
            return tmp

        def pre(src, dst):
            if (src.id > dst.id):
                return (dst, src)
            else:
                return (src, dst)

        def post(src, dst, path):
            if src.id > dst.id:
                return path.reverse()
            else:
                return path

        self.allowRecoveryPaths = True
        self.pathsSortedStatically = ReducibleLazyEvaluation(initlizar, pre, post)

    def prepare(self):
        assert self.topo.isClean()

    def P4Adaptive(self):
        visited = set(list(filter(lambda x: x.swapped(), self.topo.links)))
        for _, width, p in self.pathsSortedDynamically:
            nodes = copy.deepcopy(p)
            oldNumOfPairs = len(self.topo.getEstablishedEntanglements(p[0], p[-1]))
            pendingInbound = {node: set() for node in p}
            pendingOutbound = {node: set() for node in p}
            for i in range(1, width + 1):
                segmentsToTry = []
                segmentsToTry.append((0, len(p) - 1))
                while segmentsToTry is not []:
                    si, di = segmentsToTry.pop(0)
                    # for this part I think it's just create a topo instead of function of Node
                    src, dst = nodes[si], nodes[di]
                    links = list(filter(lambda x: x.assigned,
                                        list(self.topo.kHopNeighbors(src).intersection(self.topo.kHopNeighbors(dst)))))
                    #               for this part I'm not sure
                    edges = list(map(lambda x: x, links.groupby_dict_(links, lambda x: to(x.n1, x.n2))))
                    rp = self.topo.shortestPath(edges, src, dst, self.fStateMetric)
                    if rp[1] is []:
                        mi = (si + di) / 2
                        if mi != si:
                            segmentsToTry.append((si, di))
                        if mi != di:
                            segmentsToTry.append((mi, di))
                    else:
                        while (True):
                            elinks = list(filter(lambda x: x.entangled and x not in visited, links))
                            edges = set(list(map(lambda x: to(x.n2, x.n2), elinks)))
                            _, rp = self.shortestPath(edges, src, dst, self.fStateMetric)
                            if rp is None:
                                break
                            #     n12 [node,node]
                            #     next node
                            for n12, next in list(zip(list(zip(rp[0:-2], rp[1:-1])), rp[2:])):
                                prev, n = n12
                                prevLink = \
                                sorted(list(filter(lambda x: x.entangled and (prev in x) and (not x.swappedAt(n))
                                                             and (True if x in pendingInbound[prev] else False),
                                                   n.links)),
                                       key=lambda x: x.id)[0]

                                nextLink = \
                                sorted(list(filter(lambda x: x.entangled and (next in x) and (not x.swappedAt(n))
                                                             and (True if x in pendingInbound[next] else False),
                                                   n.links)),
                                       key=lambda x: x.id)[0]
                                if prevLink == None or nextLink == None:
                                    continue
                                visited.add(prevLink)
                                visited.add(nextLink)

                                if prev == rp[0] and not prevLink.swappedAt(prev):
                                    pendIn = list(sorted(pendingInbound[prev], key=lambda x: x.id))
                                    pin = removeUntil(pendIn, lambda x: not x.swappedAt(prev))
                                    if pin != None:
                                        prev.attemptSwapping(pin, prevLink)
                                    else:
                                        pendingOutbound[prev].add(prevLink)

                                if next == rp[-1] and not nextLink.swappedAt(next):
                                    pendOut = list(sorted(pendingInbound[next], key=lambda x: x.id))
                                    pout = removeUntil(pendOut, lambda x: not x.swappedAt(next))
                                    if pout != None:
                                        prev.attemptSwapping(pout, nextLink)
                                    else:
                                        pendingOutbound[next].add(nextLink)
                                    n.attemptSwapping(prevLink, nextLink)
                                if len(rp) == 2:
                                    prev, next = rp
                                    pendIn = list(sorted(pendingInbound[prev], key=lambda x: x.id))
                                    pendOut = list(sorted(pendingInbound[next], key=lambda x: x.id))
                                    link = \
                                    sorted(list(filter(lambda x: x.entangled and next in x and not (x.s1 and x.s2)
                                                                 and x not in pendIn and x not in pendOut and x not in visited,
                                                       prev.links)))[0]
                                    if link:
                                        visited.add(link)
                                        pin = removeUntil(pendIn, lambda x: not x.swappedAt(prev))
                                        if pin and not link.swappedAt(prev):
                                            prev.attemptSwapping(pin, link)
                                        else:
                                            pendingOutbound[prev].add(link)
                                        pout = removeUntil(pendOut, lambda x: not x.swappedAt(next))
                                        if pout and not link.swappedAt(next):
                                            next.attemptSwapping(pout, link)
                                        else:
                                            pendingInbound[next].add(link)
            succ = len(self.topo.getEstablishedEntanglements(p[0], p[-1])) - oldNumOfPairs
            self.logWriter.write(f"{list(map(lambda x: x.id, p))} " + f"{width} {succ} " + "//offline")

    def P2(self):
        self.pathsSortedStatically.clear()
        self.extraPaths.clear()

        allPaths = [self.pathsSortedStatically[it] for it in self.srcDstPairs]

        # this queue needs to have tuples in it with the tup[0] being the actual priority
        # https://www.pythonpool.com/python-priority-queue/
        # I think it is okay to trade some space off for getting comparators out of the way.
        # The logic can be implemented when we push the elements to the queue with a function call to get the priority of the element

        q, type = PriorityQueue(), self.priorityType
        for path in allPaths:
            q.push(path, getPriority(path, self.priorityType))

        cnt = 0

        while q and cnt < 100 * len(self.srcDstPairs):
            p = q.pop()
            width = self.topo.widthPhase2(p[3])

            if width == 0:
                q.push(p, getPriority(p, self.priorityType))
                break

            if width < p[1]:
                newP = (p[0], width, p[2])
                # didn't add the priority type before
                q.push(newP, getPriority(newP, self.priorityType))
                continue

            self.pathsSortedDynamically.append(p)

            for i in range(1, width + 1):
                for node1, node2 in Path.edges(p[2]):
                    for link in node1.links:
                        if (link.node1 == node2 or link.node2 == node2) and link.assigned:
                            link.assignQubits()
                            break

        self.P2Online(q)

        if self.allowRecoveryPaths:
            self.P2Recovery(q)

    def P2Recovery(self, q):
        pass

    def P2Online(self, q):
        pass


class SumDist(OfflinePathBasedAlgorithm):
    def __init__(self, topo, allowRecoveryPaths):
        super().__init__(topo, allowRecoveryPaths)
        self.name = 'SumDist-R' if allowRecoveryPaths else 'SumDist'
        self.priorityType = 'SumDist'


class CreationRate(OfflinePathBasedAlgorithm):
    def __init__(self, topo, allowRecoveryPaths):
        super().__init__(topo, allowRecoveryPaths)
        self.name = 'CR-R' if allowRecoveryPaths else 'CR'
        # this lambda function is the input to initialize ReducibleLazyEvaluation
        self.fStateMetric = ReducibleLazyEvaluation(
            lambda link: math.exp(self.topo.alpha * length([link.node1.loc, link.node2.loc])))
        self.priorityType = 'CR'


class MultiMetric(OfflinePathBasedAlgorithm):
    def __init__(self, topo, allowRecoveryPaths):
        super().__init__(topo, allowRecoveryPaths)
        self.name = 'MultiMetric-R' if allowRecoveryPaths else 'MultiMetric'
        self.priorityType = 'MultiMetric'


class BotCap(OfflinePathBasedAlgorithm):
    def __init__(self, topo, allowRecoveryPaths):
        super().__init__(topo, allowRecoveryPaths)
        self.name = 'BotCap-R' if allowRecoveryPaths else 'BotCap'
        self.priorityType = 'BotCap'


# this class is not used anywhere
class GlobalLinkState():
    pass
