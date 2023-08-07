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
        pq = MinHeap()
        shortest_route_length = self.shortest_path_BFS(source, dst)
        if shortest_route_length == -1:
            return
        update_graph = self.topo
        D_pur = defaultdict(lambda: 0)

        # for Hmin: E|C|:
        for min_cost in range(shortest_route_length, len(self.purification_table.keys())*max([len(k) for k in self.purification_table.values()])):
            paths = self.k_shortest_paths(source, dst, reqs)
            for path in paths:
                cost = 0
                path_fidelity = self.calc_path_fidelity(path)
                while path_fidelity < self.threshold:
                    link = self.min_fidelity_link(path) # Identify link with minimum fidelity to purify
                    D_pur[link] += 1
                    cost += 1
                    self.purification_table[link] = self.purification_table[link][1:]
                    path_fidelity = self.calc_path_fidelity(path) # Since calc path fid is based off table, updates in table affect fidelity
                cost += len(path) - 1 # Num entanglements used for swapping
                pq.push(cost, path, D_pur)
            # path = pq.pop()
            # while path[0] <= min_cost + 1:
            #     if self.path_width(path, D_pur) >= 1:
                    # for edge in path:
                    #   edge - min(path_width, Reqs) - num_purification on the edge (from D_pur)
            #     path = pq.pop()
        return pq



    def path_width(self, path, D_pur):
        # Finds Wmin(i, j) defined in pg7
        min_width = 0
        n1 = n2 = path[0]
        for i in range(len(path) - 1):
            min_width = min(min_width, len(self.purification_table[(path[i], path[i+1])])/(D_pur[n1, n2] + 1))
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
    
    def calc_path_fidelity(self, path):
        # Calculates fid through 0 purification links in pur table
        fidelity = 1
        for i in range(len(path) - 1):
            fidelity = fidelity * self.purification_table[(path[i], path[i+1])][0]
        return fidelity
                

    def min_fidelity_link(self, path):
        # Find the link with the minimum fidelity through the purification table
        min_fid = 1
        n1 = n2 = path[0]
        for i in range(len(path) - 1):
            if self.purification_table[(path[i], path[i+1])][0] < min_fid:
                min_fid = self.purification_table[(path[i], path[i+1])][0]
                n1, n2 = path[i], path[i+1]
        return (n1, n2) # Doesn't actually return link object, just two nodes
    
    # Do I need shortest path with <= k hops or k shortest paths???
    def k_shortest_paths(self, src, dst, k):
        def dijkstra(src): # Dijkstra's makes this an offline algorithm
            dist = {node: float('inf') for node in self.topo.nodes}
            dist[src] = 0
            heap = MinHeap()
            heap.push(0, src.id, src) # Will break if heap evaluates objects, so use src.id to stop all comparison ties

            while heap.get_length() > 0:
                curr_dist, curr_node_id, curr_node = heap.pop()

                if curr_dist > dist[curr_node]:
                    continue
    
                for neighbor in self.topo.kHopNeighbors(curr_node, 1):
                    distance = curr_dist + 1
                    if distance < dist[neighbor]:
                        dist[neighbor] = distance
                        heap.push(distance, neighbor.id, neighbor)
            return dist

        paths = []
        min_distances = dijkstra(src)
        if min_distances[dst] == float('inf'): # No S-D path
            return paths

        heap = MinHeap()
        i = 0
        heap.push(0, i, [src])
        while heap.get_length() > 0 and len(paths) < k:
            curr_dist, iter, curr_path = heap.pop()
            curr_node = curr_path[-1]

            if curr_node == dst:
                paths.append(curr_path)
                continue

            for neighbor in self.topo.kHopNeighbors(curr_node, 1):
                if neighbor not in curr_path:
                    i += 1
                    new_path = curr_path + [neighbor]
                    heap.push(curr_dist + 1, i, new_path)
        return paths
        
                