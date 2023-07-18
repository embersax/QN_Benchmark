# topo imports do not work in algorithms dir
from topo.Topo import Topo
from algorithm.AlgorithmBase import Algorithm
from utils.CollectionUtils import PriorityQueue
import sys

# Purification table = {
#                       (node1.id, node2.id) : [fidelity at 0 purifications..., fidelity at max purifications]
#                       (0, 1)               : [0.4, 0.7, 0.9]
#                       }

# Fidelity degration after entanglement swapping calculated by formula 4 in 
# https://arxiv.org/pdf/1906.06019.pdf#:~:text=Additionally%20for%20opera%2D%20tion%20with,pair%20of%20high%20target%20fidelity.


# Removes all links that have max fidelity < required threshold
# Since fidelity list created in increasing order, fid_list[-1] returns max fidelity
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
        if (link.n1.id, link.n2.id) not in purification_table.keys(): 
            purification_table[(link.n1.id, link.n2.id)] = [link.fidelity]
        else:
            # Apply Formula to find max purified fidelity
            x1, x2 = link.fidelity, purification_table[link.n1.id, link.n2.id][-1]
            purification_table[(link.n1.id, link.n2.id)].append(x1*x2/(x1*x2 + (1-x1)*(1-x2)))
    return purification_table



class QPath():
    def __init__(self, topo, threshold):
        self.topo = topo
        self.threshold = threshold
        self.purification_table = remove_lower_threshold(populate_purification_table({}, self.topo), self.threshold)
        self.name = 'QPath'
    def P2(self, source, dst):
        self.q = PriorityQueue()
        self.source = source
        self.dst = dst
        shortest_route_length = self.shortest_path_BFS(self.source, self.dst)
        if shortest_route_length == -1:
            return
        update_graph = self.topo
        for min_hops in range(shortest_route_length, sys.maxsize):
            self.purification_table

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
    
    # def path_fidelity(path, purification_decisions):
    #     fidelity = 0
    #     for node in path:
            
            