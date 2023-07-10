# Returns fidelity of link after n rounds of purification
# Assumes entanglement success on several links between two nodes is equal


import os
print(os.listdir('topo'))

from topo.Topo import Topo

netTopology = Topo.generateString(30, 0.6, 5, 0.1, 6)
topo = Topo(netTopology)
print(Topo.links)
def purify(x1, x2, rounds):
    if rounds == 0:
        return max(x1, x2)
    purified_fidelity = x1*x2/(x1*x2 + (1-x1)*(1-x2))
    return purify(purified_fidelity, x2, rounds - 1)

def prune(Topo, fidelity_threshold):
    links = Topo.links
    for link in links:
        if purify(link.probEntanglementSuccess(), link.probEntanglementSuccess()) > fidelity_threshold:
            Topo.remove(link)

