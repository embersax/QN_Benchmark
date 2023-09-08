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



# Returns the fidelity after a single purification
def purify(f1, f2):
    return f1*f2/(f1*f2 + (1-f1)*(1-f2))

def sort_link(n1, n2):
    if n1.id > n2.id:
        return (n2, n1)
    return (n1, n2)


# Returns the path given a 
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
        fid, parents = self.extended_dijkstra(src)
        if fid[dst] == 0: # No path
            return
        path = find_path(parents, dst)
        average_fid = self.threshold**(1/len(path))

        path2 = self.topo.shortestPathYenAlg(src, dst, 1, [list(link) for link in self.purification_table.keys()])
        print(path2[0][0])
        print(path)
        assert len(path2[0][0]) == len(path)
        return average_fid

    def extended_dijkstra(self, src): 
        fid = {node: 0 for node in self.topo.nodes}
        parents = {node: None for node in self.topo.nodes}
        fid[src] = 1
        heap = MinHeap()
        heap.push(1, src.id, src) # Will break if heap evaluates objects, so use src.id to stop all comparison ties

        while heap.get_length() > 0:
            curr_fid, curr_id, curr_node = heap.pop()

            if curr_fid < fid[curr_node]:
                continue
            
            for link in curr_node.links:
                fidelity = curr_fid*link.fidelity
                neighbor = link.n2 if link.n1.id == curr_id else link.n1
                if fidelity > fid[neighbor]:
                    fid[neighbor] = fidelity
                    heap.push(fidelity, neighbor.id, neighbor)
                    parents[neighbor] = curr_node
        return (fid, parents)