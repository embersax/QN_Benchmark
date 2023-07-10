# topo imports do not work in algorithms dir
from topo.Topo import Topo

# Purification table = {
#                       (node1.id, node2.id) : [fidelity at 0 purifications..., fidelity at max purifications]
#                       (0, 1)               : [0.4, 0.7, 0.9]
#                       }


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
        # Apply Formula to find max purified fidelity
        else:
            x1, x2 = link.fidelity, purification_table[link.n1.id, link.n2.id][-1]
            purification_table[(link.n1.id, link.n2.id)].append(x1*x2/(x1*x2 + (1-x1)*(1-x2)))
    return purification_table



class QPath():
    def __init__(self, topo, threshold):
        self.topo = topo
        self.threshold = threshold
        self.purification_table = remove_lower_threshold(populate_purification_table({}, self.topo), self.threshold)
        self.name = 'QPath'

