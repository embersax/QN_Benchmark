# topo imports do not work in algorithms dir
from topo.Topo import Topo
from algorithm.AlgorithmBase import Algorithm
from utils.CollectionUtils import MinHeap
from collections import Counter, defaultdict
import heapq

# Purification table = {
#                       (node1, node2) : [fidelity at 0 purifications..., fidelity at max purifications]
#                       (node id: 0, node id: 1)               : [0.4, 0.7, 0.9]
#                       }

# Path tracing credits to https://stackoverflow.com/questions/8922060/how-to-trace-the-path-in-a-breadth-first-search

# Returns the fidelity after a single purification
def purify(f1, f2):
    return f1*f2/(f1*f2 + (1-f1)*(1-f2))

def sort_link(n1, n2):
    if n1.id > n2.id:
        return (n2, n1)
    return (n1, n2)

def find_neighbors(node):
    neigh = set()
    for link in node.links:
        neigh.add(link.n1)
        neigh.add(link.n2)
    return list(neigh)

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
        shortest_route_length = self.shortest_path_BFS(source, dst)
        if shortest_route_length == -1:
            return
        update_graph = self.topo

        # for Hmin: E|C|:
        # for min_cost in range(shortest_route_length, len(self.purification_table.keys())*max([len(k) for k in self.purification_table.values()])):
        for min_cost in range(1):
            pq = MinHeap()
            paths = self.topo.shortestPathYenAlg(source, dst, reqs)
            for i in range(len(paths)):
                path = paths[i][0]
                cost = paths[i][1]
                D_pur = defaultdict(lambda: 0)
                path_fidelity = self.calc_path_fidelity(path, D_pur)
                print(f"Original path fid: {path_fidelity}")
                while path_fidelity < self.threshold:
                    link = self.min_fidelity_link(path, D_pur) # Identify link with minimum fidelity to purify
                    print(f"current link: {link}")
                    if link[0] == link[1]: # No possible purifications
                        break
                    D_pur[link] += 1
                    cost += 1
                    path_fidelity = self.calc_path_fidelity(path, D_pur) 
                    print(f"updated path fidelity: {path_fidelity}")
                pq.push(cost, D_pur, path) # Cost won't always be unique
            # route = pq.pop() # (cost, path, D_pur)
            # while pq.get_length() > 0 and route[0] <= min_cost + 1:
            #     path_width = self.calc_path_width(route)
            #     if path_width >= 1:
            #         for link in route[1]:
            #             print(link)
            #             # has_capacity(edge, cap) and has_memory(edge, mem)
            #             # link - min(path_width, reqs) - num_purification on the edge (from D_pur)
            #     route = pq.pop()
        while pq.get_length() > 0:
            print(pq.pop())
        return pq

    def has_memory(link, mem):
        # T/F if link nodes have nQubits
        return min(link.n1.nQubits, link.n2.nQubits) >= mem
    
    def calc_path_width(self, route):
        # Finds Wmin(i, j) defined in pg7
        min_width = 0
        n1 = n2 = route[0]
        for i in range(len(route) - 1):
            min_width = min(min_width, len(self.purification_table[(route[i], route[i+1])])/(route[2][n1, n2] + 1))
        return min_width
        
                
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
    
    def calc_path_fidelity(self, path, D_pur):
        fidelity = 1
        for i in range(len(path) - 1):
            link = sort_link(path[i], path[i+1])
            fidelity = fidelity * self.purification_table[link][D_pur[link]]
        return fidelity
                

    def min_fidelity_link(self, path, D_pur):
        # Find the link with the minimum fidelity through the purification table
        min_fid = 1
        n1 = n2 = path[0]
        for i in range(len(path) - 1):
            link = sort_link(path[i], path[i+1])
            if D_pur[link] >= len(self.purification_table[link]) - 1: # No more possible purifications on link
                continue
            link_fid = self.purification_table[link][D_pur[link]]
            if link_fid < min_fid:
                min_fid = link_fid
                n1, n2 = path[i], path[i+1]
        return sort_link(n1, n2) # Doesn't actually return link object, just two nodes
    