from collections import defaultdict
from copy import deepcopy

from algorithm.AlgorithmBase import Algorithm
from topo.Topo import Topo, Path
from topo.Node import to
import numpy as np
import heapq
from utils.utils import ReducibleLazyEvaluation
import abc
from dataclasses import dataclass
from functools import reduce
from itertools import dropwhile
from algorithm.AlgorithmBase import Algorithm

import shortestpaths as sp

# Mainly P2 changes (maximize served quantum-user pairs and expected throughput)
class MultiEntanglement(Algorithm):
    def __init__(self, topo, allowRecoveryPaths=False):
        super().__init__(topo)
        self.allowRecoveryPaths = allowRecoveryPaths
        self.name="Multi_E"
        # An element in the list is: [(s_node, d_node), shortestPaths]
        # Where shortestPathsX is: [([s_node.id, path_node1.id, ..., d_node.id], cost), (...)]
        self.shortestPathsForPairs = []
         # This is a list of PickedPaths
        self.majorPaths = []
        #  HashMap<PickedPath, LinkedList<PickedPath>>()
        self.recoveryPaths = {}
        self.pathToRecoveryPaths = {}
    
    # Sort each S-D pair list of paths by cost in ascending order
    def sortPathsByCost(self):
        for i in range(0, len(self.shortestPathsForPairs)):
            self.shortestPathsForPairs[i][1].sort(key=lambda x: x[1])
    
    # Maximize source-destination pairs
    # Select main routing path for each pair
    # Mind Switchs' qubits assigned for paths
    def selectivePaths(self):
        expectedPathNum = len(self.srcDstPairs)
        retrievedPathNum = pow(expectedPathNum, 2)
        print("expectedPathNum: " + str(expectedPathNum) + " retrievedPathNum: " + str(retrievedPathNum))
        for pair in self.srcDstPairs:
            src, dst = pair[0], pair[1]
            shortestPaths = self.topo.shortestPathYenAlg(src.id, dst.id, retrievedPathNum)
            self.shortestPathsForPairs.append([pair, shortestPaths])
        print(str(self.shortestPathsForPairs))
        print("Sort paths by shortest to longest: ")
        # Sort paths by shortest to longest
        self.sortPathsByCost()
        print(str(self.shortestPathsForPairs))
        
        # For each S-D pair, check if it has the correct number of paths
        for i in range(0, len(self.srcDstPairs)):
            pathNum = len(self.shortestPathsForPairs[i][1])
            if pathNum < expectedPathNum:
                # Add paths until S-D pair has M paths
                # Note: for smaller topologies, there wouldn't be other unique paths to add
                print("Need to add paths")
            elif pathNum > expectedPathNum:
                print("Need to delete " + str(pathNum - expectedPathNum) + " paths")
                # Delete largest cost paths until M paths
                for extraPath in range(0, pathNum - expectedPathNum):
                    self.shortestPathsForPairs[i][1].pop(-1)
        print("Final Result: ")
        print(str(self.shortestPathsForPairs))
    
    def branchAndPrice1(self):
        # Given current path, the selected path and ?
        # new trimmed list of selectivePaths() = empty
        # for pair, paths in self.shortestPathsForPairs:
        #     if paths without trim anded with self.shortestPathsForPairs and self.shortestPathsForPairs is possible:
        #         add path to new trimmed path
        # if the length of new trimmed list <=1
        #     mark S-D pair
        #     if the length of new trimmed list ==1:
        #         ? = 1
        #     Compare ? and solution, update solution if needed
        #     find
    
    # Given a list of paths for S-D pairs with selectivePaths(),
    # select one main shortest distance path for each S-D pair
    def integerSolution1(self):
        # Set selected main path of S-D pair as 0
        # Set path signal of all paths for each S-D pair as 0
        # Sort the list of paths from selectivePaths() in descending order
        # for pair, paths in self.shortestPathsForPairs:
        #     if ? (might be path from selectivePaths) == 1:
        #         mark S-D pair by setting path signal as 1?
        
        # Find max path from selectivePaths() <1 and satisfies s-d not entangled
        #self.branchAndPrice1()
    
    # def branchAndPrice2(self):
    
    # Maximize expected throughput of all source-destination pairs from Step1
    # Determine qubits assigned to paths from Step1
    # Mind Switchs' qubits assigned for paths
    # def integerSolution2(self):
    
    # Currently using OnlineAlgorithm's functions to run topology
    def prepare(self):
        pass

    def P2(self):
        assert self.topo.isClean()
        self.majorPaths.clear()
        self.recoveryPaths.clear()
        # IMPORTANT: This is not implemented yet. It uses the lay evaluation function.
        self.pathToRecoveryPaths.clear()
        while True:
            candidates = self.calCandidates(self.srcDstPairs)
            if candidates is not None and len(candidates) > 0:
                pick = max(candidates,key=lambda x: x[0])
                if pick is not None and pick[0] > 0.0:
                    pick = (pick[0], pick[1], tuple(pick[2]))
                    self.pickAndAssignPath(pick)
            else:
                break
    # ops is a list of pairs of nodes
    # This function returns a list of picked path: a triple
    def calCandidates(self, ops):
        candidates = []
        for o in ops:
            src, dst = o[0], o[1]
            maxM = min(src.remainingQubits, dst.remainingQubits)
            if maxM == 0: return None
            # Candidate should be of type PickedPath, which is a (Double,Int,Path) triple.
            candidate = None
            # In the kotlin code it goes until 1 but, to include 1, we must set the range parameter to 0
            for w in range(maxM, 0, -1):
                a, b, c = set(self.topo.nodes), set([src]), set([dst])
                # We subtract the intersection from the union to get the difference between the three sets.
                tmp = a-b-c
                # In the kotlin code, it's a hashset, but I think a set works fine.
                failNodes = set([node for node in tmp if node.remainingQubits < 2 * w])
                tmp0 = [link for link in self.topo.links if
                        not link.assigned and link.n1 not in failNodes and link.n2 not in failNodes]
                tmp1 = {}
                for link in tmp0:
                    if to(link.n1,link.n2) in tmp1:
                        tmp1[to(link.n1,link.n2)].append(link)
                    else:
                        tmp1[to(link.n1,link.n2)] = [link]
                # So I think we do not need the filter part if we do it this way. We only need the edges.
                edges = set([edge for edge in tmp1 if len(tmp1[edge]) >= w])
                # TODO: ReducibleLazyEvaluation part
                # key: node, value: list of nodes
                neighborsOf = defaultdict(list)
                for edge in edges:
                    neighborsOf[edge.n1].append(edge.n2)
                    neighborsOf[edge.n2].append(edge.n1)


                # if neighborsOf == {}: continue

                if src not in neighborsOf.keys() or dst not in neighborsOf.keys() : continue
                # This is a hashmap of nodes: <Node,Node>
                prevFromSrc = {}

                def getPathFromSrc(n):
                    # This is a list of nodes
                    path = []
                    cur = n
                    while cur != self.topo.sentinal:
                        path.insert(0, cur)
                        cur = prevFromSrc[cur]
                    return path

                def priority(edge):
                    node1, node2 = edge.n1, edge.n2
                    if E[node1.id][0] < E[node2.id][0]:
                        return 1
                    elif E[node1.id][0] == E[node2.id][0]:
                        return 0
                    else:
                        return -1
                E = [(float('-inf'), [0.0] * (w + 1)) for _ in self.topo.nodes]
                q = []

                heapq.heappush(q, (-E[src.id][0], to(src, self.topo.sentinal)))
                E[src.id] = (float('inf'), [0.0] * (w + 1))
                prevFromSrc = {}
                while q:
                    # if(w==1):
                        # print("w=1")
                    _, eg = heapq.heappop(q)
                    u, prev = eg.n1, eg.n2
                    if u in prevFromSrc:  # skip same node suboptimal paths
                        continue
                    prevFromSrc[u] = prev  # record
                    if u == dst:
                        candidate = E[u.id][0], w, getPathFromSrc(dst)
                        candidates.append(candidate)
                        break
                    for neighbor in neighborsOf[u]:
                        tmp = deepcopy(E[u.id][1])
                        e = self.topo.e(getPathFromSrc(u) + [neighbor], w, tmp)
                        newE = e, tmp
                        oldE = E[neighbor.id]

                        if oldE[0] < newE[0]:
                            E[neighbor.id] = newE
                            try:
                                heapq.heappush(q, (-newE[0], to(neighbor, u)))
                            except TypeError as e:
                                # print(-newE[0])
                                print(src.id, dst.id)

                                for element in q:
                                    print("current width:  " +str(w))
                                    print(element[0],element[1].n1.id,element[1].n2.id)



        return [c for c in candidates if c is not None]
    # The pick variable is a (Double, Int, Path) triple. Recall that a path is a list of nodes.
    def pickAndAssignPath(self, pick, majorPath=None):
        if majorPath is not None:
            self.recoveryPaths[majorPath] = pick
        else:
            self.majorPaths.append(pick)
            self.recoveryPaths[pick] = []
        # We get the second value of the triple
        width = pick[1]
        # The first argument of toAdd is a link bundle: list of links. The third argument is a map, where the key is of
        # type Edge and the values are lists of pairs. The pairs are made of connections (lists of link bundles).
        toAdd = ([], width, {})
        tmp_edge = [to(pick[2][i], pick[2][i + 1]) for i in range(len(pick[2]) - 1)]

        for edge in tmp_edge:
            n1, n2 = edge.n1, edge.n2
            links = sorted([link for link in n1.links if not link.assigned and link.contains(n2)],
                           key=lambda link: link.id)[:width]
            assert len(links) == width
            toAdd[0].append(links)
            for link in links:
                link.assignQubits()
                link.tryEntanglement()  # just for display

    def P4(self):

        for pathWithWidth in self.majorPaths:
            _, width, majorPath = pathWithWidth
            oldNumPairs = len(self.topo.getEstablishedEntanglements(majorPath[0], majorPath[-1]))
            recoveryPaths = sorted(self.recoveryPaths[pathWithWidth],
                                   key=lambda tup: len(tup[2]) * 10000 + majorPath.index(tup[2][0]))

            for _, w, p in recoveryPaths:
                available = min(
                    [len([link.contains(node2) and link.entangled for link in node1.links]) for node1, node2 in
                     Path.edges(p)])
                self.pathToRecoveryPaths[pathWithWidth].append(RecoveryPath(p, w, 0, available))

            edges = list(zip(range(len(majorPath) - 1), range(1, len(majorPath))))
            rpToWidth = {recPath[2]: recPath[1] for recPath in recoveryPaths}

            for i in range(1, width + 1):

                def filterForBrokenEdges(tup):
                    i1, i2 = tup
                    n1, n2 = majorPath[i1], majorPath[i2]
                    checkAny = [link.contains(n2) and link.assigned and link.notSwapped() for link in n1.links]
                    return any(checkAny)

                brokenEdges = list(filter(filterForBrokenEdges, edges))
                edgeToRps = {brokenEdge: [] for brokenEdge in brokenEdges}
                rpToEdges = {recPath[2]: [] for recPath in recoveryPaths}

                for _, _, rp in recoveryPaths:
                    s1, s2 = majorPath.index(rp[0]), majorPath.index(rp[-1])
                    reqdEdges = list(
                        filter(lambda edge: edge in brokenEdges, list(zip(range(s1, s2), range(s1 + 1, s2 + 1)))))

                    for edge in reqdEdges:
                        rpToEdges[rp] = edge
                        edgeToRps[edge] = rp

                realPickedRps = {}
                realRepairedEdges = {}

                # Try to cover the broken edges

                for brokenEdge in brokenEdges:
                    if brokenEdge in realRepairedEdges:
                        continue
                    repaired = False
                    next = 0

                    tryRpContinue = False
                    for rp in list(sorted(list(
                            filter(lambda it: rpToWidth[it] > 0 and not it in realPickedRps, edgeToRps[brokenEdge])),
                                          key=lambda it: majorPath.index[it[0]] * 10000 + majorPath.index[it[-1]])):
                        # there is only a single for loop in this case. So, I don't think the labeled continue makes a difference.
                        if majorPath.index[rp[0]] < next:   continue
                        next = majorPath.index[rp[-1]]
                        repairedEdges = set(realRepairedEdges)
                        pickedRps = set(realPickedRps)

                        otherCoveredEdges = set(rpToEdges[rp]).difference(brokenEdge)

                        for edge in otherCoveredEdges:
                            prevRpSet = set(edgeToRps[edge]).intersection(set(pickedRps)).remove(rp)
                            prevRp = prevRpSet[0] if prevRpSet else None

                            if prevRp == None:
                                repairedEdges.add(edge)

                            else:
                                continue

                        repaired = True
                        repairedEdges.add(brokenEdge)
                        pickedRps.add(rp)

                        for item1, item2 in zip(realPickedRps, pickedRps):
                            item = item1 - item2
                            rpToWidth[item] += 1
                            item = -item
                            rpToWidth[item] -= 1

                        realPickedRps = pickedRps
                        realRepairedEdges = repairedEdges
                        break

                    if not repaired:
                        break

                def doInFold(acc, rp):
                    idx = -1
                    for ele in self.pathToRecoveryPaths[pathWithWidth]:
                        if ele.path == rp:
                            idx = self.pathToRecoveryPaths[pathWithWidth].index(ele)

                    pathData = self.pathToRecoveryPaths[pathWithWidth][idx]
                    pathData.taken += 1
                    toAdd = Path.edges(rp)
                    toDelete = Path.edges(list(
                        dropwhile(lambda it: it != rp[-1],
                                  list(reversed(list(dropwhile(lambda it: it != rp[0], acc)))))))
                    edgesOfNewPathAndCycles = set(Path.edges(acc)).difference(toDelete).union(toAdd)

                    # our implementation of ReducibleLazyEvaluation requires 2 inputs K and V but Shoqian initialized it with just one. Has to be looked at.
                    p = self.topo(edgesOfNewPathAndCycles, acc[0], acc[-1], ReducibleLazyEvaluation(1.0))
                    return p

                def foldLeft(realPickedRps, majorPath):
                    return reduce(doInFold, realPickedRps, majorPath)

                p = foldLeft(realPickedRps, majorPath)
                zippedP = list(zip(list(zip(p[:-2],p[1:-1])),p[2:]))
                for n12, next in zippedP:
                    prev, n = n12
                    prevLinks = list(sorted(filter(lambda it: it.entangled and not it.swappedAt(n) and  not it.utilized and it.contains(prev), n.links),
                                            key=lambda it: it.id))
                    nextLinks = list(sorted(filter(lambda it: it.entangled and not it.swappedAt(n) and  not it.utilized and it.contains(next), n.links),
                                            key=lambda it: it.id))
                    if prevLinks ==[] or nextLinks ==[]:
                        continue

                    prevAndNext = list(zip([prevLinks[0]], [nextLinks][0]))
                    for l1, l2 in prevAndNext:
                        n.attemptSwapping(l1, l2)
                        l1.utilize()
                        if(next == p[-1]):
                            l2.utilize()
            succ =0
            if(len(majorPath) >2):
                succ = len(self.topo.getEstablishedEntanglements(majorPath[0], majorPath[-1])) - oldNumPairs
                self.established.append(((p[0], p[-1]), succ))
            else:
                SDlinks = sorted([link for link in majorPath[0].links if
                                  link.entangled and not link.swappedAt(majorPath[0]) and link.contains(
                                      majorPath[-1]) and not link.utilized], key=lambda x: x.id)

                if SDlinks:
                    succ = min(len(SDlinks), width)
                    for pid in range(succ):
                        SDlinks[pid].utilize()
                    self.established.append(((p[0], p[-1]), succ))

            self.logWriter.write("""{}, {} {}""".format([it.id for it in majorPath], width, succ))
            # for it in self.pathToRecoveryPaths[pathWithWidth]:
            #     self.logWriter.write(
            #         """{}, {} {} {}""".format([it2.id for it2 in it.path], width, it.available, it.taken))

        self.logWriter.write("\n")
