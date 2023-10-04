# topo imports do not work in algorithms dir
from topo.Topo import Topo
from algorithm.AlgorithmBase import Algorithm
from utils.CollectionUtils import MinHeap
from collections import Counter, defaultdict
import sys
import heapq

# Purification table = {
#                       (node1, node2) : [fidelity at 0 purifications..., fidelity at max purifications]
#                       ...
#                       }
# All links are ordered to lookup in pure table and pure decisions 

# Path tracing credits to https://stackoverflow.com/questions/8922060/how-to-trace-the-path-in-a-breadth-first-search



# Returns the fidelity after a single purification
def purify(f1, f2):
    return f1*f2/(f1*f2 + (1-f1)*(1-f2))

# Doesn't actually deal with links, just orders a tuple of nodes
def sort_link(n1, n2):
    if n1.id > n2.id:
        return (n2, n1)
    return (n1, n2)

# Returns the path given a list of parent nodes
def find_path(d, dst):
    path = [dst]
    back = d[dst]
    while back:
        path.insert(0, back)
        back = d[back]
    return path

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

class Route():
    def __init__(self, cost, path, pur_dec):
        self.cost = cost
        self.path = path
        self.pur_dec = pur_dec
    def __lt__(self, other):
        return self.cost < other.cost
    def __repr__(self):
        return f"\n\nCost: {self.cost}\nPath: {self.path}\nPurification Decisions: {self.pur_dec}"

class QLeap():
    def __init__(self, topo, threshold):
        self.topo = topo
        self.threshold = threshold
        self.purification_table = remove_lower_threshold(populate_purification_table({}, self.topo), self.threshold)
        self.name = 'QPath'
    def P2(self, src, dst, reqs):
        update_graph = self.purification_table
        sol_paths = []
        throughput = 0

        for r in range(reqs):

            # Pathfinding
            fid, parents = self.extended_dijkstra(src)
            if fid[dst] == 0: # No path
                return sol_paths
            path = find_path(parents, dst)
            if len(path) == 1: # src == dst
                return [src]

            # Purification decisions
            average_fid = self.threshold**(1/(len(path)-1))
            D_pur = defaultdict(lambda: 0)
            for i in range(len(path) - 1):
                link = sort_link(path[i], path[i+1])
                D_pur[link] = self.min_pur(link, average_fid)
            
            # Resource alloc
            route = Route(len(path) + sum(D_pur.values()) - 1, path, D_pur)
            path_width = self.calc_path_width(route)
            if path_width >= 1:
                sol_paths.append(route)
                for i in range(len(route.path) - 1):
                    link = sort_link(route.path[i], route.path[i+1])
                    num_usable = min(self.num_memory(link)//route.cost, self.num_capacity(link)//route.cost)
                    path_width = min(path_width, num_usable)

                # Mark links as used
                for i in range(len(route.path) - 1):
                    link = sort_link(route.path[i], route.path[i+1])
                    to_mark = path_width
                    while to_mark > 0:
                        for l in self.topo.links:
                            if l.n1.id == link[0].id and l.n2.id == link[1].id:
                                l.utilized = True
                                to_mark -= 1

           
            self.purification_table = update_graph            
            throughput += path_width
            sol_paths.append((route, path_width))

            if throughput >= reqs:
                return sol_paths

        return sol_paths

    def num_memory(self, link):
        # T/F if link nodes have nQubits
        return min(link[0].nQubits, link[1].nQubits)

    def num_capacity(self, link):
        # fact or cap: A link can support cap capacity
        return len(self.purification_table[link])

    # returns the min number of purifications on a link to reach threshold fidelity
    def min_pur(self, link, f):
        for i in range(len(self.purification_table[link])):
            if self.purification_table[link][i] >= self.threshold:
                return i
        print('Something WONGGGGGG(ook yi)')
        sys.exit(1)
    
    def calc_path_width(self, route):
        # Finds Wmin(i, j) (pg7)
        max_width = float('inf')
        for i in range(len(route.path) - 1):
            capacity = len(self.purification_table[sort_link(route.path[i], route.path[i+1])])
            cost = route.pur_dec[sort_link(route.path[i], route.path[i+1])] + 1
            max_width = min(max_width, capacity//cost)
        return max_width

    def extended_dijkstra(self, src): 
        fid = {node: 0 for node in self.topo.nodes}
        max_fid = {node: 1 for node in self.topo.nodes}
        parents = {node: None for node in self.topo.nodes}
        fid[src] = 1
        heap = MinHeap()
        heap.push(1, src.id, src) # Will break if heap evaluates objects, so use src.id to stop all comparison ties

        while heap.get_length() > 0:
            curr_fid, curr_id, curr_node = heap.pop()

            if curr_fid < fid[curr_node]:
                continue
            
            for link in curr_node.links:
                if not link.utilized:
                    fidelity = curr_fid*link.fidelity
                    neighbor = link.n2 if link.n1.id == curr_id else link.n1
                    max_fid[neighbor] = max_fid[neighbor] * self.purification_table[sort_link(curr_node, neighbor)][-1]
                    if fidelity > fid[neighbor] and max_fid[neighbor] >= self.threshold:
                        fid[neighbor] = fidelity
                        heap.push(fidelity, neighbor.id, neighbor)
                        parents[neighbor] = curr_node
        return (fid, parents)