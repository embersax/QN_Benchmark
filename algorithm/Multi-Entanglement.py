"""
This class is based on the Multi-Entanglement Routing Design over Quantum Networks research paper
by Yiming Zeng, Jiarui Zhang, Ji Liu, Zhenhua Liu, and Yuanyuan Yang. 
(Includes OnlineAlgorithm functions for debugging purposes with main.py)

Two dataclasses (SDpairInfo and pathInfo) are created for readability and organization when storing
source destination pairs, their corresponding paths, and other information.

Note: This paper assumes that the source and destination nodes are not used as switch nodes for other
pairs. Therefore, the qubit resource constraint ignores source and destination nodes.
"""

# Packages only OnlineAlgorithm requires
from collections import defaultdict
from copy import deepcopy
from topo.Node import to
import numpy as np
import heapq
from utils.utils import ReducibleLazyEvaluation
import abc
from functools import reduce
from itertools import dropwhile

# Packages both OnlineAlgorithm and MultiEntanglement requires
from algorithm.AlgorithmBase import Algorithm
from topo.Topo import Topo, Path
from dataclasses import dataclass
from algorithm.AlgorithmBase import Algorithm

# Packages only MultiEntanglement requires
import shortestpaths as sp
import cplex

@dataclass
class pathInfo:
    pathNodes: list[int] = None  # A list of numbers representing node id numbers
    cost: int = None             # The cost or length of the path as specified by pathNodes
    selected: float = 0.0        # Represents if the path was selected for entanglement (by maximizeUserPairs function)

@dataclass
class SDpairInfo:
    SDpair: tuple                # Stores 2 node items
    paths: list[pathInfo] = None
    marked: bool = False         # Marked by Algorithm 2 (Integer Solution Recovery) in the research paper at line 5

# Mainly P2 changes (maximize served quantum-user pairs and expected throughput)
class MultiEntanglement(Algorithm):
    def __init__(self, topo, allowRecoveryPaths=False):
        super().__init__(topo)
        self.name="Multi_E"
        self.allSDpairInfo = []  # A list of SDpairInfo
        self.majorPaths = []     # A list of PickedPaths

        # Based on OnlineAlgorithm's fields
        self.allowRecoveryPaths = allowRecoveryPaths
        self.recoveryPaths = {}  # HashMap<PickedPath, LinkedList<PickedPath>>()
        self.pathToRecoveryPaths = {}
    
    # Helper functions 
    # Sort each SD pair list of paths by cost in ascending order if order is False
    def sortPathsByCost(self, order=False):
        for i in range(0, len(self.allSDpairInfo)):
            self.allSDpairInfo[i].paths.sort(key=lambda x: x.cost, reverse=order)
    
    # Converts nodeID to Node object
    def nodeIDtoNode(self, nodeID):
        for node in self.topo.nodes:
            if nodeID == node.id:
                return node

    # Find all used switch ID numbers from all pathNodes in allSDpairInfo
    # Returns a list of Node objects
    def findSwitches(self):
        s = []
        for sdInfo in self.allSDpairInfo:
            for path in sdInfo.paths:
                s = list(set(path.pathNodes[1:-1]) | set(s))

        # Convert into Node objects
        switches = []
        for switchID in s:
            switches.append(self.nodeIDtoNode(switchID))

        # print(s, "\n")
        # print(switches, "\n")
        return(switches)

    # Algorithm 1 Selective Paths Algorithm
    # Find and sort (in ascending length) shortest distance paths of each SD pair
    # Stores result in self.allSDpairInfo
    def selectivePaths(self):
        expectedPathNum = len(self.srcDstPairs)
        retrievedPathNum = pow(expectedPathNum, 2)
        # print("expectedPathNum: " + str(expectedPathNum) + " retrievedPathNum: " + str(retrievedPathNum))
        
        for pair in self.srcDstPairs:
            src, dst = pair[0], pair[1]
            shortestPaths = self.topo.shortestPathYenAlg(src.id, dst.id, retrievedPathNum)
            self.allSDpairInfo.append(SDpairInfo(pair, [pathInfo(pNodes, cost) for pNodes, cost in shortestPaths]))
        # print("\n" + str(self.allSDpairInfo))
        # print("\nSort paths by shortest to longest: ")
        
        # Sort paths by shortest to longest
        self.sortPathsByCost()
        # print(str(self.allSDpairInfo))
        
        # Remove longest distance paths of all paths list until length of all paths list = (number of SD pairs)^2
        while sum(len(pairInfo.paths) for pairInfo in self.allSDpairInfo) > retrievedPathNum:
            # print("Total amount of paths: " + str(sum(len(pairInfo.paths) for pairInfo in self.allSDpairInfo)))
            deleteInfo = None
            # Find longest path to be deleted
            for pairInfo in self.allSDpairInfo:
                if len(pairInfo.paths) > expectedPathNum:
                    if (deleteInfo == None) or (pairInfo.paths[-1].cost > deleteInfo.paths[-1].cost):
                        deleteInfo = pairInfo
            # print("Deleted: ", str(deleteInfo.paths[-1]))
            deleteInfo.paths.pop(-1)
            
        # For each SD pair, check if it has the correct number of paths
        for pairInfo in self.allSDpairInfo:
            pathNum = len(pairInfo.paths)
            
            # Add paths until SD pair has M paths (unimplemented)
            # Note: for smaller topologies, there wouldn't be other unique paths to add
            # if pathNum < expectedPathNum:
                # print("Need to add paths")
            if pathNum > expectedPathNum:
                # print("Need to delete " + str(pathNum - expectedPathNum) + " paths")
                # Delete largest cost paths until M paths
                for extraPath in range(0, pathNum - expectedPathNum):
                    pairInfo.paths.pop(-1)
        
        # print("Final Result: ")
        # print(str(self.allSDpairInfo))
        self.maximizeUserPairs()

    # Solving Problem S1 with Cplex (and preparing for Algorithm 2)
    # Maximize source-destination pairs and select main routing path for each pair
    # Adds in constraints for limited qubit resource and variable representing if route is selected
    def maximizeUserPairs(self):
        switchesList = self.findSwitches()
        
        # Create optimization model and set problem statement to be maximized
        opt_mod = cplex.Cplex()
        opt_mod.objective.set_sense(opt_mod.objective.sense.maximize)
        
        # For each path, add decision variables, selectedState (which is a value from 0 to 1), to the problem statement with a coefficient of 1
        # For each SD pair, create an expression for the constraint to make the sum of all selectedState for the pair's paths <= 1
            # Notes: ind is the decision variable while val is the coefficient the corresponding decision variable is multiplied with
            # The constraint expression = (ind0*val0) + (ind1*val1) + ...
        for pairNum, pairInfo in enumerate(self.allSDpairInfo):
            expr1 = cplex.SparsePair(ind=[], val=[])
            for pathNum, pInfo in enumerate(pairInfo.paths):
                opt_mod.variables.add(obj=[1.0], lb=[0], ub=[1], names=["selectedState" + str(pairNum) + "." + str(pathNum)])
                expr1.ind.append("selectedState" + str(pairNum) + "." + str(pathNum))
                expr1.val.append(1.0)
            opt_mod.linear_constraints.add(lin_expr=[expr1], senses=["L"], rhs=[1])

        # Add constraint: For any switch (meaning nodes between SD and not SD nodes themselves), their total qubits must be enough to support all selected paths using the switch
            # Note: This program assumes that SD pairs do not act as switches for others (all switches are honest)
        # Iterating through each switch, find the number of times it is used in each selected path (the minimum amount of qubits for the switch is 2 times of that amount)
        for node in switchesList:
            # print("Node: " + str(node.id) + " RemainingQubits: " + str(node.remainingQubits))
            capacity = int(node.remainingQubits/2)
            expr2 = cplex.SparsePair(ind=[], val=[])
            for pairNum, pairInfo in enumerate(self.allSDpairInfo):
                for pathNum, pInfo in enumerate(pairInfo.paths):
                    expr2.ind.append("selectedState" + str(pairNum) + "." + str(pathNum))
                    expr2.val.append(len(set(pInfo.pathNodes[1:-1]) & set([node.id])))
            opt_mod.linear_constraints.add(lin_expr=[expr2], senses=["L"], rhs=[capacity])

        opt_mod.solve()
        # print("Number of SD pairs serviced  = ", opt_mod.solution.get_objective_value())

        # Transfer solution over to allSDpairInfo structures
        numcols = opt_mod.variables.get_num()
        sol = opt_mod.solution.get_values()
        i = 0
        for pairInfo in self.allSDpairInfo:
            for pInfo in pairInfo.paths:
                pInfo.selected = sol[i]
                i += 1
                # if int(pInfo.selected) == 1:
                #     print(pInfo)
        self.integerSolution1()

    # Unimplemented
    def branchAndPrice1(self, curPair, curPath):
        # print(curPair, curPath, curPath.selected)
        return
    
    # Given a list of paths for SD pairs with selectivePaths() and maximizeUserPairs(),
    # select one main shortest distance path for each SD pair
    def integerSolution1(self):
        # Sort the list of paths from selectivePaths() in descending order
        self.sortPathsByCost(True)
        
        # Mark SD pairs with clear selected route (when selected == 1)
        for pairInfo in self.allSDpairInfo:
            for path in pairInfo.paths:
                if int(path.selected) == 1:
                    pairInfo.marked = True
        # print(self.allSDpairInfo)
        # Find maximum selected value that is < 1 and its SD pair is not entangled
        curPair, curPath, maxSelected = None, None, None
        for pairInfo in self.allSDpairInfo:
            if pairInfo.marked is False:
                for path in pairInfo.paths:
                    if ((maxSelected == None) or (path.selected > maxSelected)) and (path.selected < 1):
                        maxSelected = path.selected
                        curPair = pairInfo
                        curPath = path

        if maxSelected is not None:
            self.branchAndPrice1(curPair, curPath)
    
    # OnlineAlgorithm's functions below
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
