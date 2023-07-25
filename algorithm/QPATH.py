# topo imports do not work in algorithms dir
from topo.Topo import Topo
from algorithm.AlgorithmBase import Algorithm
from utils.CollectionUtils import PriorityQueue
import sys

# Purification table = {
#                       (node1, node2) : [fidelity at 0 purifications..., fidelity at max purifications]
#                       (node id: 0, node id: 1)               : [0.4, 0.7, 0.9]
#                       }

# Fidelity degration after entanglement swapping calculated by formula 4 in 
# https://arxiv.org/pdf/1906.06019.pdf#:~:text=Additionally%20for%20opera%2D%20tion%20with,pair%20of%20high%20target%20fidelity.

# Path tracing credits to https://stackoverflow.com/questions/8922060/how-to-trace-the-path-in-a-breadth-first-search

# Returns the fidelity after an entanglement swapping
def swap_fidelity(f):
    return f**2 + ((1-f)**2)/3

# Returns the fidelity after a single purification
def purify(f1, f2):
    return f1*f2/(f1*f2 + (1-f1)*(1-f2))

# Removes all links that have max fidelity < required threshold
def remove_lower_threshold(purification_table, threshold):
    keys_to_del = []
    for link, fid_list in purification_table.items():
        if fid_list[-1] < threshold:
            keys_to_del.append(link)
    for key in keys_to_del:
        del purification_table[key]
    return purification_table

# Creates purification table from given topology
def populate_purification_table(purification_table, topo):
    for link in topo.links:
        if (link.n1, link.n2) not in purification_table.keys(): 
            purification_table[(link.n1, link.n2)] = [link.fidelity]
        else:
            # Apply Formula to find max purified fidelity
            x1, x2 = link.fidelity, purification_table[link.n1, link.n2][-1]
            purification_table[(link.n1, link.n2)].append(purify(x1, x2))
    return purification_table

class QPath():
    def __init__(self, topo, threshold):
        self.topo = topo
        self.threshold = threshold
        self.purification_table = remove_lower_threshold(populate_purification_table({}, self.topo), self.threshold)
        self.name = 'QPath'
    def P2(self, source, dst, reqs):
        self.q = PriorityQueue()
        shortest_route_length = self.shortest_path_BFS(self.source, self.dst)
        if shortest_route_length == -1:
            return
        update_graph = self.topo
        path_fidelity = self.calc_path_fidelity(self.returns_shortest_path(source, dst))
        # for min_hops in range(shortest_route_length, len(self.purification_table.keys())*max([len(k) for k in self.purification_table.values()])):
        #     while path_fidelity < self.threshold:
        #         path_fidelity = path
                
    def shortest_path_BFS(self, source, dst):
        # returns min distance by num hops between two nodes
        # returns -1 if node unreachable
        dist = {source: 0}
        q = [source]
        while len(q) > 0:
            node = q.pop()
            if node == dst:
                return dist[node]
            for link in node.links:
                if link.n2 not in dist.keys():
                    q.append(link.n2)
                    dist[link.n2] = dist[source] + 1
                else:
                    dist[link.n2] = min(dist[source] + 1, dist[link.n2])
        return -1
    
    def returns_shortest_path(self, source, dst):
        # Returns the path in list form [node, node]
        queue = [(source,[source])]
        visited = set()

        while queue:
            vertex, path = queue.pop(0)
            visited.add(vertex)
            print(vertex)
            # Getting nodes from the links, link.n2 is an adjacent node
            for link in vertex.links:
                if link.n2 == dst:
                    return path + [dst]
                else:
                    if link.n2 not in visited:
                        visited.add(link.n2)
                        queue.append((link.n2, path + [link.n2]))
        return queue
    def calc_path_fidelity(self, path):
        fidelity = 1
        for i in range(len(path) - 1):
            fidelity = fidelity * self.purification_table[(path[i], path[i+1])][0]
        return fidelity
                
    def min_fidelity_link(self, path):
        min_fid = 1
        n1 = n2 = path[0]
        for i in range(len(path) - 1):
            if min_fid < self.purification_table[(path[i], path[i+1])][0]:
                min_fid = self.purification_table[(path[i], path[i+1])][0]
                n1 = path[i]
                n2 = path[i+1]
        return (n1, n2)
            
            