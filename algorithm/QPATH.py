# topo imports do not work in algorithms dir
from topo.Topo import Topo
from algorithm.AlgorithmBase import Algorithm
from utils.CollectionUtils import MinHeap
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

        return self.k_shortest_paths(source, dst, 3)
        # for Hmin: E|C|:
        # for min_cost in range(shortest_route_length, len(self.purification_table.keys())*max([len(k) for k in self.purification_table.values()])):
        #     cost = 0
        #     D_pur = [] # Purification Decisions
        #     paths = [] # List of paths with cost = min_cost
        #     path = self.returns_shortest_path(source, dst)
        #     path_fidelity = self.calc_path_fidelity(path)
        #     while path_fidelity < self.threshold:
        #         link = self.min_fidelity_link(path) # Identify link with minimum fidelity to purify
        #         D_pur.append(link)
        #         cost += 1
        #         self.purification_table[link] = self.purification_table[link][1:]
        #         path_fidelity = self.calc_path_fidelity(path) # Since calc path fid is based off table, updates in table affect fidelity
        #     cost += len(path) - 1 # Num entanglements used for swapping
        #     pq.push(cost, path, D_pur)

            
            
        
                
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
        # Find the link with the minimum fidelity through the purification table
        min_fid = 1
        n1 = n2 = path[0]
        for i in range(len(path) - 1):
            if self.purification_table[(path[i], path[i+1])][0] < min_fid:
                min_fid = self.purification_table[(path[i], path[i+1])][0]
                n1 = path[i]
                n2 = path[i+1]
        return (n1, n2) # Doesn't actually return link object, just two nodes
    
    def k_shortest_paths(self, src, dst, k):
        def dijkstra(src):
            dist = {node: float('inf') for node in self.topo.nodes}
            dist[src] = 0
            heap = [(0, src)]

            while heap:
                curr_dist, curr_node = heapq.heappop(heap)

                if curr_dist > dist[curr_node]:
                    continue
    
                for neighbor in self.topo.kHopNeighbors(curr_node, 1):
                    distance = curr_dist + 1
                    if distance < dist[neighbor]:
                        dist[neighbor] = distance
                        heapq.heappush(heap, (distance, neighbor))
            return dist

        paths = []
        min_distances = dijkstra(src)
        if min_distances[dst] == float('inf'):
            return paths

        heap = []
        heapq.heappush(heap, (0, [src]))

        while heap and len(paths) < k:
            curr_dist, curr_path = heapq.heappop(heap)
            curr_node = curr_path[-1]

            if curr_node == dst:
                paths.append(curr_path)
                continue

            for neighbor in self.topo.kHopNeighbors(curr_node, 1):
                if neighbor not in curr_path:
                    new_path = curr_path + [neighbor]
                    heapq.heappush(heap, (curr_dist + 1, new_path))
        return paths
        
                
                