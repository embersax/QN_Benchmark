import math
import hashlib
import sys
import re
from itertools import combinations, groupby
from topo.Node import Node, Edge,to
from topo.Link import Link
from utils.Disjoinset import Disjointset
from utils.utils import *
from utils.CollectionUtils import PriorityQueue
from collections import defaultdict
import shortestpaths as sp
import networkx as nx
import random

hopLimit = 15


def priority(n1, n2):
    if n1.id < n2.id:
        return -1
    elif n1.id > n2.id:
        return 1
    else:
        return


class Path:
    """
    A class for all the methods called as Path.<methodname>()
    All methods in this class are static and are solely utilitarian
    """

    @staticmethod
    def edges(path):
        return [Edge(node1, node2) for (node1, node2) in zip(path[:len(path) - 1], path[1:])]

    @staticmethod
    def applyCycle():
        # There isn't any call made to this function. If we see a need for this, it can be implemented then.
        pass


class Topo:

    def __init__(self, input):
        # self.n = 0
        self.alpha = 0
        self.q = 0
        self.k = 0
        self.nodes = []
        self.links = []
        self.sentinal = None
        self.nodeDigits = 0
        self.linkDigits = 0
        self.distanceDigits = 0
        self.internalLength = 0
        # determine  type of input,if input type is string continue, if is Topo object, get to string
        if isinstance(input, Topo):
            input = input.toString()
        x = input.splitlines()
        x_next = list(map(lambda y: re.sub("""\s*//.*$""", "", y), x))
        lines = list(filter(lambda x: x != "", x_next))
        n = int(lines.pop(0))
        # Why do we have -sys.maxint and not sys.maxint ?
        ## Python 3 has no maxint. Replacing it with maxsize
        self.sentinal = Node(self, -1, [-1, -1, -1], sys.maxsize)
        self.alpha = float(lines.pop(0))
        self.q = float(lines.pop(0))
        self.k = int(lines.pop(0))
        # In kotlin, div by zero apparently is totally fine. So, changed the internalLength computation by a bit.
        self.internalLength = math.log(1 / self.q) / self.alpha if self.alpha != 0 else float('inf')

        for i in range(n):
            line = lines.pop(0)
            # I think we need it to be float(x)
            tmp = list(map(lambda x: float(x), line.split(" ")))
            node = Node(self, i, [tmp[1], tmp[2]], int(tmp[0]))
            self.nodes.append(node)

        self.nodeDigits = round(math.ceil(math.log10(float(len(self.nodes)))))

        while len(lines):
            linkLine = lines.pop(0)
            # I think it should be linkLine.split(" ") instead of line.split(" ")
            tmp = list(map(lambda x: int(x), linkLine.split(" ")))
            tmp1 = tmp[0:2]
            tmp2 = list(map(lambda x: self.nodes[x], tmp1))
            # if(len(tmp2)==1):
                # print(tmp2)
            # I made a tuple with n1 and n2, as Shouqian did.
            (n1, n2) = sorted(tmp2, key=lambda x: x.id)
            # I think it has to be tmp, as tmp1[2] is None.
            nm = tmp[2]
            assert n1.id < n2.id
            # I'm not sure but maybe as range is not inclusive for the upper limit, it has to be range(1,nm+1),
            # according to Shouqian's code.
            for i in range(1, nm + 1):
                link = Link(self, n1, n2, length(list_minus(n1.loc, n2.loc)))
                self.links.append(link)
                n1.links.append(link)
                n2.links.append(link)

        self.linkDigits = round(math.ceil(math.log10(len(self.links))))
        flat_map = lambda f, xs: (y for ys in xs for y in f(ys))

        self.distanceDigits = round(math.ceil(math.log10(
            max(list(map(lambda x: x.l, self.links)) +
                list(map(lambda x: abs(x), list(flat_map(lambda x: x.loc, self.nodes))))))))

    # All links, entanglements and swappings are set to false.
    @property
    def n(self):
        return len(self.nodes)
    def clearEntanglements(self):
        for link in self.links:
            link.clearEntanglement()
            link.swap1, link.swap2 = False, False

        for node in self.nodes:
            node.internalLinks.clear()
            assert node.nQubits == node.remainingQubits

    # We only clear the swappings in phase 4, but not the entanglements.
    def clearOnlyPhase4(self):
        for link in self.links:
            link.swap1, link.swap2 = False, False

        for node in self.nodes:
            node.internalLinks.clear()

    # This is the implementation of the shortestPath algorithm
    # We first need to take all edges (in both directions)
    def shortestPath(self, edges, src, dst, fStateMetric):
        ## a step-by-step decomposition of Shoqian's neighborsOf val

        # reversedEdges = [Edge(edge.n2, edge.n1) for edge in edges]
        # allEdges = [edge.toList() for edge in reversedEdges]  # this currently has reversedEdges only
        # allEdges = allEdges.extend(edges)
        # allEdgesList = allEdges.toList()
        # connectedNodesViaEdges = groupby(allEdgesList, key=lambda x: x[0])
        # neighborsOf = [Edge(key, Edge(val[1], fStateMetric[Edge(val[0], val[1])])) for key, val in connectedNodesViaEdges.items()]
        # # I'm having a lot of trouble implementing the neighborsOf part of the code-
        # neighborsOf = set(neighborsOf)

        ## Trying to make it a one-liner hoping that it will make better use of space

        neighborsOf = set([Edge(key, Edge(val[1], fStateMetric[Edge(val[0], val[1])])) for key, val in groupby(
            [edge.toList() for edge in [Edge(edge.n2, edge.n1) for edge in edges]].extend(edges).toList(),
            key=lambda x: x[0]).items()])
        prevFromSrc = {}
        D = dict.fromkeys(self.nodes, float('inf'))
        # Do we need to have a comparator? Can we just use a def
        q = PriorityQueue()
        D[src.id] = 0.0
        q.push(src.nodeTo(self.sentinal), priority(src, self.sentinal))
        while q:
            # Not sure about this pop, but I think it is correct based on the comparator
            (w, prev) = q.pop()
            if w in prevFromSrc: continue
            prevFromSrc[w] = prev

            if w == dst:
                path = []
                cur = dst
                while cur != self.sentinal:
                    path.insert(0, cur)
                    cur = prevFromSrc[cur]
                return D[dst.id], path

            for p in neighborsOf[w]:
                neighbor, weight = p[0], p[1]
                newDist = D[w.id] + weight
                oldDist = D[neighbor.id]
                if oldDist > newDist:
                    D[neighbor.id] = newDist
                    q.push(neighbor.nodeTo(w), priority(neighbor, w))
        # I don't understand this return statement.
        return float('inf'), []

    # Returns all routes for two nodes
    def getAllRoutes(self, n1_, n2_):
        n1, n2 = [min(n1_, n2_), max(n1_, n2_)]
        linksAndEdges = {}
        for link in self.links:
            linksAndEdges[link] = link.n1.nodeTo(link.n2)
        topoStr = '\n'.join(str(el) for el in set(linksAndEdges))
        digest = hashlib.sha256()
        digest.update(bytearray(topoStr.encode()))
        # I'm not sure what to do with the routeStorage part
        result = []
        range_ = self.kHopNeighbors(self.nodes[n1], (hopLimit + 1) / 2).add(self.kHopNeighbors(self.nodes[n2],
                                                                                             (hopLimit + 1) / 2))
        # If result is empty
        if not result:
            # Find all
            def dfs(l, remainingNeighbors):
                if l[-1] == n2:
                    result.append(list(l))
                elif len(l) > hopLimit:
                    return
                else:
                    filtered = filter(lambda it: it in range_, self.nodes[l[-1]].neighbors)
                    map_ = {}
                    for node in filtered:
                        map_[node] = node.id
                    for _, v in map_:
                        if v not in l and remainingNeighbors:
                            l.append(v)
                            dfs(l, list(set(remainingNeighbors) - set(self.nodes[v])))
                            l.pop(len(l) - 1)

            dfs(list(n1), list(set(self.nodes[n2].neighbors) - set(self.nodes[n1])))

            # Sort via Dijkstra
            # I'm not sure about this part
            p = {}
            return result

    @staticmethod
    def listtoString(s, seperator):
        s = list(map(lambda x: str(x), s))
        return seperator.join(s)

    def toString(self):
        return self.toConfig()

    @staticmethod
    def groupby_dict(s, f):
        tmp = {}
        for a, b in groupby(s, f):
            for thing in b:
                tmp[a] = b
        return tmp

    def toConfig(self):
        nl = "\n"
        tmp_dict = groupby_dict(self.links, lambda x: to(x.n1,x.n2))
        grouped_links = {}

        # for link in self.links:
        #     # Create a tuple from n1 and n2 to use as a dictionary key
        #     # key = (link.n1, link.n2)
        #     key = to(link.n1, link.n2)
        #
        #     if key not in grouped_links:
        #         # If the key doesn't exist in the dictionary, create a new list
        #         grouped_links[key] = [link]
        #     else:
        #         # If the key does exist, append the link to the existing list
        #         grouped_links[key].append(link)
        # print(len(tmp_dict))
        return f"""{self.n}
{self.alpha}
{self.q}
{self.k}
{self.listtoString(list(map(lambda node: str(node.nQubits) +" "+ self.listtoString(node.loc, " "), self.nodes)), nl)}
{self.listtoString(list(map(lambda x: str(x.n1.id)+" " + str(x.n2.id)+" " +str(len(tmp_dict[x])), tmp_dict)), nl)}
"""

    def toFullString(self):
        nl = "\n"
        tmp_dict = self.grouby_dict(self.links, lambda x: x.n1.to(x.n2))
        return f"""
{self.n}
{self.alpha}
{self.q}
{self.k}
{self.listtoString(list(map(lambda node: str(node.remainingQubits) + "/" + str(node.nQubits) + self.listtoString(node.loc, " "), self.nodes)), nl)}
{self.listtoString(list(map(lambda x: str(x.n1.id) + str(len(tmp_dict[x])) + str(len(tmp_dict[x])), tmp_dict)), nl)}
"""

    def widthPhase2(self, path):
        tmp1 = min(list(map(lambda x: x.remainingQubits / 2, path[1:-1])))
        p1 = tmp1 if tmp1 else sys.maxsize
        p2 = min(list(
            map(lambda x: sum(1 for link in x[0].links if ((link.n1 == x[1] | link.n2 == x[1]) & (not link.assigned))),
                list(zip(path[:-1], path[1:])))))
        return min(path[0].remainingQubits, path[-1].remainingQubits, p1, p2)

    def widthPhase4(self, path):

        return min(list(map(lambda x: sum(
            1 for link in x[0].links if ((link.n1 == x[1] | link.n2 == x[1]) & link.entangled & link.notSwapped())),
                            list(zip(path[:-1], path[1:])))))

    def e(self, path, mul, oldP):
        s = len(path) - 1
        P = [0.0 for _ in range(mul + 1)]
        tmp_edge = [Edge(path[i], path[i+1]) for i in range(len(path)-1)]
        p = [0.0] + list(map(lambda x: math.exp(-self.alpha * length(list_minus(x.n1.loc, x.n2.loc))), tmp_edge))
        start = s

        ## Not sure what 'm', 'C_' are. Worth checking again.

        if sum(oldP) == 0.0:
            for i in range(0, mul+1):
                # print(i)
                oldP[i] = C_(mul, i) * p[1] ** (i) *(1 - p[1]) ** (mul - i)
            start = 2
        assert len(oldP) == mul + 1

        for k in range(start, s+1):
            for i in range(0, mul+1):
                exactlyM = C_(mul, i) * p[k] ** (i) * (1 - p[k]) ** (mul - i)
                atLeastM = sum(list(map(lambda x: C_(mul, x) * p[k] ** (x) * (1 - p[k]) ** (mul - x),
                                        [l for l in range(i + 1, mul+1)]))) + exactlyM
                tmp = 0
                for l in range(i + 1, mul+1):
                    tmp += oldP[l]
                P[i] = oldP[i] * atLeastM + exactlyM * tmp
            for i in range(0, mul+1):
                oldP[i] = P[i]

        assert abs(sum(oldP) - 1.0) < 0.0001

        return sum(list(map(lambda x: x * oldP[x], [i for i in range(1, mul+1)]))) * self.q ** (s - 1)


    # Returns all routes between two nodes
    def getAllElementCycles(self):
        linksAndEdges = {}
        for link in self.links:
            linksAndEdges[link] = link.node1.nodeTo(link.node2)
        topoStr = '\n'.join(str(el) for el in set(linksAndEdges))
        digest = hashlib.sha256()
        digest.update(bytearray(topoStr))
        # I'm not sure what to do with the routeStorage part
        result = []
        # If result is empty
        if not result:
            resultSet = []
            for n in self.nodes:
                # Not sure but maybe it has to be +1 to be inclusive
                for length in range(3, 10 + 1):
                    def dfs(l):
                        tmp = l[-1].neighbors.intersection(l)
                        if len(l) == length:
                            if tmp == set(l[l.size - 2], n):
                                tmp1 = {}
                                for node in l: tmp1[node] = node.id
                                l = tmp1
                                m = l.index(min(l))
                                resultSet.append(l[m:len(l)] + l[0:m])
                        elif len(tmp) <= 1:
                            filtered = filter(lambda it: len(l) == 1 or it != l[len(l) - 2], l[-1].neighbors)
                            for f in filtered:
                                l.append(f)
                                dfs(l)
                                l.pop(len(l) - 1)

                    dfs(list(n))
            result.extend(resultSet)
            self.saveRoutes()
        # I don't understand this return statement
        return dict.fromkeys(result, self.nodes)

    def getStatistics(self):

        # I don't think we necessarily need to sort each and every one of these lists. Just the min() and the max()
        # will do. We can remove these once we are trying to optimize. For now, I'm doing exactly how the kotlin one
        # does.

        numLinks = sorted([len(node.links) for node in self.nodes])
        numNeighbors = sorted([len([link.otherThan(node) for link in node.links]) for node in self.nodes])
        linkLengths = sorted([(link.node1.loc - link.node2.loc) for link in self.links])
        linkSuccPossibilities = sorted([pow(math.e, -self.alpha + linkLength) for linkLength in linkLengths])
        numQubits = sorted([node.nQubits for node in self.nodes])

        avgLinks = sum(numLinks) / self.n
        avgNeighbors = sum(numNeighbors) / self.n
        avgLinkLength = sum(linkLengths) / len(self.links)
        avglinkSuccP = sum(linkSuccPossibilities) / len(self.links)
        avgQubits = sum(numQubits) / self.n

        # I didn't understand why the .format() is used here. I've rounded them off to the decimal places provided in
        # the parameter S: ! I'm not sure that this string formatting is correct in Python. The $ is used in Kotlin
        # to introduce variables values
        return f"""
              Topology:
              {self.n} nodes, {len(self.links)} links         alpha: {self.alpha}  q: {self.q}
              #links     per node                (Max, Avg, Min): {numLinks[-1]}                        {round(avgLinks, 2)}        {numLinks[0]}
              #qubits    per node                (Max, Avg, Min): {numQubits[-1]}                           {round(avgQubits, 2)}       {numQubits[0]}
              #neighbors per node                (Max, Avg, Min): {numNeighbors[-1]}                    {round(avgNeighbors, 2)}    {numNeighbors[0]} 
              length of links       (km)         (Max, Avg, Min): {round(linkLengths[-1], 2)}               {round(avgLinkLength, 2)}   {round(linkLengths[0], 2)} 
              P(entanglement succeed for a link) (Max, Avg, Min): {round(linkSuccPossibilities[-1], 2)} {round(avglinkSuccP, 2)}    {round(linkSuccPossibilities[0], 2)}
              """

    # Function to get the k-hop neighbors in the network.
    def kHopNeighbors(self, root, k):
        # In case we have a bigger k than the given in the initial conditions, we return all the nodes.
        if k > self.k: return set(self.nodes)
        # A list of false values with the length of nodes list.
        registered = [False for _ in self.nodes]
        # We create a stack and we push the root. We also set the root as a visited node.
        stack = [root]
        registered[root.id] = True

        def work():
            # We take the current node of the stack.
            current = stack[-1]
            # We go recursively through the stack if its size is equal or smaller than k+1.
            if len(stack) <= k + 1:
                # From the current node, we take all the nodes it has links with. We then take the ones that are not yet
                # registered and build a set out of them.
                unregisteredNeighbors = set(filter(lambda x: x.id not in registered,
                                                   [Edge(link.n1, link.n2).otherThan(current) for link in
                                                    current.links]))

                for unregisteredNeighbor in unregisteredNeighbors:
                    # We check the neighbor as visited.
                    registered[unregisteredNeighbor.id] = True
                    # We push the neighbor to the stack so that we can then evaluate the neighbor's neighbors.
                    stack.append(unregisteredNeighbor)
                    # We call recursively this function.
                    work()
                    # We pop the current node from the stack.
                    stack.pop()

        work()
        # We return a list of the k neighbors.
        res = []
        for idx, val in enumerate(registered):
            if val:
                res.append(self.nodes[idx])
            else:
                res.append(self.sentinal)

        res = set(filter(lambda x: x != self.sentinal, res))

        return res

    # It does the same as the previous function, but instead of returning a list of the k-neighbors, it returns all the
    # links in the range.
    def kHopNeighborLinks(self, root, k):
        registered = [False for _ in self.nodes]
        result = set()

        stack = [root]
        registered[root.id] = True

        def work():
            current = stack[-1]
            result.union(self.current.links)

            if len(stack) <= k + 1:
                unregisteredNeighbors = set(filter(lambda x: x.id not in registered,
                                                   [Edge(link.node1, link.node2).otherThan(current) for link in
                                                    current.links]))

                for unregisteredNeighbor in unregisteredNeighbors:
                    registered[unregisteredNeighbor.id] = True
                    stack.append(unregisteredNeighbor)
                    work()
                    stack.pop()

        work()

        return result

    # Function to get the established entanglements on the topology.
    def getEstablishedEntanglements(self, node1, node2):
        # We create a stack of pairs: <link, node>
        # We push n1 to the stack.
        stack = [(None, node1)]
        # Result is a list of paths, which are lists of nodes.
        result = []

        while stack:
            incoming, current = stack.pop()
            # We create a list with just one element: n2
            if current == node2:
                path = [node2]
                ### As there may be an error, maybe we can use a try-catch
                try:
                    inc = incoming
                except:
                    raise RuntimeError('No such incoming node')
                # For the incoming link, while its n1 and n2 aren't the n1 we are studying, we take the previous node.
                # The previous node is either n2 (if the link's n1 is the last of the path) or n1 otherwise.
                # We update inc to be the first (next) internal link that is not inc.
                while inc.n1 != node1 and inc.n2 != node2:
                    prev = inc.n2 if inc.n1 == path[-1] else inc.n1
                    # inc = [link.otherThan(inc) for link in prev.internalLinks if inc in link]
                    # inc = [otherThan(link,inc) for link in prev.internalLinks if inc in link]
                    tmp = list(filter(lambda x:inc in x , prev.internalLinks))
                    inc = otherThan(list(filter(lambda x:inc in x , prev.internalLinks))[0],inc)
                    path.append(prev)
                # We add n1 to the path and reverse what we found
                path.append(node1)
                result.append(list(reversed(path)))
                continue
            # The if case happens in the first iteration. We take all the links that are not entangled/swapped.
            # The second case chooses all links that contain the incoming one and maps them to the other side.
            outgoingLinks = list(filter(lambda link: link.entangled and not link.swappedAt(current),
                                        current.links)) if incoming is None else [otherThan(link,incoming) for link in
                                                                                  list(filter(lambda internalLink:
                                                                                              incoming in internalLink,
                                                                                              current.internalLinks))]
            # We push every link to the stack to be evaluated.
            # The node that is pushed is the "other side" of current.
            for outgoingLink in outgoingLinks:
                stack.append([outgoingLink, Edge(outgoingLink.n1, outgoingLink.n2).otherThan(current)])

        return result

    def isClean(self):
        areLinksClean, areNodesClean = True, True
        for link in self.links:
            areLinksClean = areLinksClean and not link.entangled and not link.assigned and link.notSwapped()
        for node in self.nodes:
            areNodesClean = areNodesClean and not node.internalLinks and node.nQubits == node.remainingQubits

        return areLinksClean and areNodesClean

    # This also has not been called anywhere
    def linksBetween(self, node1, node2):
        return list(filter(lambda link: node2 == link.node1 or node2 == link.node2, [link for link in node1.links]))

    def shortestPathYenAlg(self, source, destination, numberOfPaths, links):
            edgeAndNodes = []
            for link in links:
                # How to determine edges? Using recorded or entangled links? How to double check its correctness?
                # if link.entangled is True:
                #     print("Entangled: ", str(link.n1.id), str(link.n2.id))
                if link not in edgeAndNodes:
                    edgeAndNodes.append(link)
            G = nx.Graph()
            G.add_edges_from(edgeAndNodes)
            k_paths = sp.k_shortest_paths(G, source, destination, numberOfPaths, method = 'y')
            # sp.plot_paths(k_paths, G)
            return k_paths


    @staticmethod
    def generateString(n, q, k, a, degree):
        alpha = a

        ## Shoqian didn't explicitly create it in Topo.kt but it was assigned the value 100 in configs.kt
        ## Manually assigning the value. But, at a later point in time, have to move this to configs.py file
        edgeLen = 100.0
        controllingD = math.sqrt(edgeLen * edgeLen / n)
        links = []
        # #Added nodeLocs
        nodeLocs = []
        while len(nodeLocs) < n:
            ## element is a list but x is not. The '-' operator is not defined. Please recheck this part.
            # "+" operator matches the unaryPlus operator of Shouqian's code of Double Array
            element = [random.uniform(0, 1) * 100 for _ in range(2)]
            if all(length(list_minus(x, element)) > controllingD / 1.2 for x in nodeLocs):
                nodeLocs.append(element)
        nodeLocs = sorted(nodeLocs, key=lambda x: x[0] + int(x[1] * 10 / edgeLen) * 1000000)

        def argument_function(beta):
            links.clear()
            ## An efficient algorithm is already available in itertools. Replacing with combinations() call
            tmp_list = list(combinations([i for i in range(0, n)], 2))
            for i in range(len(tmp_list)):
                (l1, l2) = list(map(lambda x: nodeLocs[x], [tmp_list[i][0], tmp_list[i][1]]))
                d = length(list_minus(l2, l1))
                if d < 2 * controllingD:
                    l = min([random.uniform(0, 1) for i in range(1, 51)])
                    r = math.exp(-beta * d)
                    if l < r:
                        # to functions needed to be implemented
                        links.append([tmp_list[i][0], tmp_list[i][1]])
            tmp1 = len(links)

            return 2 * float(len(links)) / n

        # Can't fully debug unless the dynSearch and disjointSet are actually implemented
        # dynSearch needed to be implememnted,I just realized beta is not used Shouqian's code
        beta = dynSearch(0.0, 20.0, float(degree), argument_function, False, 0.2)
        # DisjoinSet needed to be implemmented
        disjoinSet = Disjointset(n)
        for i in range(len(links)):
            disjoinSet.merge(links[i][0], links[i][1])
        # compute  ccs: =(o unitil n).map.map.groupby.map.sorted
        ### finish debugging before this part
        t1 = list(map(lambda x: [x, disjoinSet.getRepresentative(x)], [i for i in range(0, n)]))
        # from shouqian's code it seems that t1 and t2 have the same transformation and t2 is basically same as t1, I'm not sure
        t2 = list(map(lambda x: [x[0], x[1]], t1))
        # [x,disjoinSet.getgetRepresentative(x)]
        t3 = groupby_dict(t2, lambda x: x[1])
        t4 = list(map(lambda x: list(map(lambda x: x[0], t3[x])), t3))
        ccs = sorted(t4, key=lambda x: -len(x))

        biggest = ccs[0]
        for i in range(1, len(ccs)):
            # this part I'm not sure since shuffle changes the order of original ccs list
            ## ccs1 is not defined. Please check
            tmp1 = random.shuffle(ccs[i])[0:3]
            for j in range(len(tmp1)):
                nearest = sorted(biggest, key=lambda x: sum(nodeLocs[x] - nodeLocs[tmp1[j]]))[0]
                tmp2 = sorted([nearest, tmp1[j]])
                links.append(Edge(tmp2[0], tmp2[1]))

        ## groupby_dict, listtoString are functions Topo class objects. They seem to be used outside the class. So, making them static in this case.
        flat_map = lambda f, xs: (y for ys in xs for y in f(ys))
        #     retrive the flatten list first
        tmp_list = list(flat_map(lambda x: [x[0], x[1]], links))
        # retriev dictionary to iterate
        tmp_dict = groupby_dict_(tmp_list, lambda x: x)

        for key in tmp_dict:
            if len(tmp_dict[key]) / 2 < 5:
                # not sure if takes the right index, needed to double check
                nearest = sorted([i for i in range(0, n)],
                                 key=lambda x: length(list_minus(nodeLocs[x], nodeLocs[key])))[
                          1:int(6 - len(tmp_dict[key]) / 2)]

                tmp_list = list(map(lambda x: [sorted([x, key])[0], sorted([x, key])[1]],
                                    nearest))
                # need to check what is added to links here
                for item in tmp_list:
                    links.append(item)

        nl = "\n"
        tmp_string1 = Topo.listtoString(
            list(map(lambda x: f"{int(random.uniform(0, 1) * 5 + 10)} "+ Topo.listtoString(x, " "), nodeLocs)), nl)
        tmp_string2 = Topo.listtoString(
            list(map(lambda x: f"{x[0]} " + f"{x[1]} " + f"{int(random.uniform(0, 1) * 5 + 3)}",
                     links))
            , nl)

        return f"""{n}
{alpha}
{q}
{k}
{tmp_string1}
{tmp_string2}
"""



    @staticmethod
    def generate(n, q, k, a, degree):
        topoStr = Topo.generateString(n, q, k, a, degree)

        return Topo(topoStr)
def otherThan(pair, element):
    if element == pair[0]:
        return pair[1]
    elif element == pair[1]:
        return pair[0]

### I added Vamsi's updated code


## Adding this part to run some basic tests on the generate() method.

if __name__ == "__main__":
    tmp_topo = Topo.generate(50, .9, 5, .1, 6)
    tmp_str= tmp_topo.toConfig()
    print(tmp_str)


