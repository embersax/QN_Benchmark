# topo imports do not work in algorithms dir
from topo.Topo import Topo
from algorithm.AlgorithmBase import Algorithm
from utils.CollectionUtils import MinHeap
from collections import Counter, defaultdict
import heapq

# Purification table = {
#                       (node1, node2) : [fidelity at 0 purifications..., fidelity at max purifications]
#                       ...
#                       }

# Path tracing credits to https://stackoverflow.com/questions/8922060/how-to-trace-the-path-in-a-breadth-first-search


class Route():
    def __init__(self, cost, path, pur_dec):
        self.cost = cost
        self.path = path
        self.pur_dec = pur_dec
    def __lt__(self, other):
        return self.cost < other.cost
    def __repr__(self):
        return f"\n\nCost: {self.cost}\nPath: {self.path}\nPurification Decisions: {self.pur_dec}"

# Returns the fidelity after a single purification
def purify(f1, f2):
    return f1*f2/(f1*f2 + (1-f1)*(1-f2))

def sort_link(n1, n2):
    if n1.id > n2.id:
        return (n2, n1)
    return (n1, n2)

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
        
        shortest_hops = self.topo.shortestPathYenAlg(source, dst, 1, [list(link) for link in self.purification_table.keys()])[0][1]
        update_graph = self.purification_table
        sol_paths = []
        # for Hmin: E|C|:
        for min_cost in range(shortest_hops, len(self.purification_table.keys())*max([len(k) for k in self.purification_table.values()])):
            pq = []
            sol_paths = []
            paths = self.topo.shortestPathYenAlg(source, dst, reqs, [list(link) for link in self.purification_table.keys()])
            # Enqueue possible paths
            for i in range(len(paths)):
                path = paths[i][0]
                cost = paths[i][1]
                D_pur = defaultdict(lambda: 0)
                path_fidelity = self.calc_path_fidelity(path, D_pur)
                while path_fidelity < self.threshold and len(pq) < reqs:
                    link = self.min_fidelity_link(path, D_pur) # Identify link with minimum fidelity to purify
                    if link == -1 or cost > min_cost + 1: # No possible purifications
                        break
                    D_pur[link] += 1
                    cost += 1
                    path_fidelity = self.calc_path_fidelity(path, D_pur) 
                if cost <= min_cost + 1 and path_fidelity >= self.threshold:
                    heapq.heappush(pq, Route(cost, path, D_pur))
            # Decide path from available resources
            throughput = 0
            while len(pq) > 0:
                route = heapq.heappop(pq) 
                if route.cost > min_cost + 1:
                    break
                path_width = self.calc_path_width(route)
                if path_width >= 1:
                    for i in range(len(route.path) - 1):
                        link = sort_link(route.path[i], route.path[i+1])
                        num_usable = min(self.num_memory(link)//route.cost, self.num_capacity(link)//route.cost)
                        if num_usable == 0:
                            continue
                        # subtract last x elements in pur table, where x = min(path_width, reqs)*num_purifications on the edge (from D_pur), aka the cost of using the route path_width times
                        self.purification_table[link] = self.purification_table[link][:num_usable*cost+1]
                sol_paths.append((route, path_width))
                throughput += path_width
                if throughput >= reqs:
                    return sol_paths
            print('check!')
            self.purification_table = update_graph
        return sol_paths

    def num_memory(self, link):
        # T/F if link nodes have nQubits
        return min(link[0].nQubits, link[1].nQubits)

    def num_capacity(self, link):
        # fact or cap: A link can support cap capacity
        return len(self.purification_table[link])
    
    def calc_path_width(self, route):
        # Finds Wmin(i, j) (pg7)
        max_width = float('inf')
        for i in range(len(route.path) - 1):
            capacity = len(self.purification_table[sort_link(route.path[i], route.path[i+1])])
            cost = route.pur_dec[sort_link(route.path[i], route.path[i+1])] + 1
            max_width = min(max_width, capacity//cost)
        return max_width
        
                
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
            if link not in self.purification_table.keys():
                print(self.topo.links)
            fidelity = fidelity * self.purification_table[link][D_pur[link]]
        return fidelity
                

    def min_fidelity_link(self, path, D_pur):
        # Find the link with the minimum fidelity through the purification table
        # Returns -1 if none available
        max_incr = max_link = 0
        for i in range(len(path) - 1):
            link = sort_link(path[i], path[i+1])
            if D_pur[link] >= len(self.purification_table[link]) - 1: # No more possible purifications on link
                continue
            if self.purification_table[link][D_pur[link]+1] - self.purification_table[link][D_pur[link]] > max_incr:
                max_link = link
                max_incr = self.purification_table[link][D_pur[link]+1] - self.purification_table[link][D_pur[link]]
        if max_incr == 0:
            return -1
        return max_link # Doesn't actually return link object, just two nodes
