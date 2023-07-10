# topo imports do not work in algorithms dir
from topo.Topo import Topo

netTopology = Topo.generateString(5, 0.6, 5, 0.1, 6)
topo = Topo(netTopology)

# Purification table = {
#                       (node1.id, node2.id) : [fidelity at 0 purifications..., fidelity at max purifications]
#                       (0, 1)               : [0.4, 0.7, 0.9]
#                       }
purification_table = {}


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

populate_purification_table(purification_table, topo)


# Check to make sure the # of links == # of possible fidelities
t = 0
for link, fid_list in purification_table.items():
    t += len(fid_list)
assert t == len(topo.links)


# def purify(x1, x2, rounds):
#     if rounds == 0:
#         return max(x1, x2)
#     purified_fidelity = x1*x2/(x1*x2 + (1-x1)*(1-x2))
#     return purify(purified_fidelity, x2, rounds - 1)


