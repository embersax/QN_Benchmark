import random

# import Topo
# from .Topo import Edge


class Node:
    # The constructor of a node. Parameters:
    # A topology topo
    # An id
    # A localization of the node in the network
    # The number of qubits, n_qubits, assigned to this node
    def __init__(self,
                 topo,
                 id,
                 loc,
                 nQubits):

        self.topo = topo
        self.id = id
        self.loc = loc
        self.nQubits = nQubits

        self.links = []
        self.internalLinks = []

        # self.neighbors = {link.otherThan(self) for link in self.links}
        self.remainingQubits = nQubits

    # add property to neighbors can avoid the error on init part when links is empty list, but I havn't done the by
    # lazy part
    @property
    def neighbors(self):
        return {link.otherThan(self) for link in self.links}

    # Attempts link-swapping inside a node
    def attemptSwapping(self, link1, link2):
        if link1.n1 == self:
            assert not link1.swap1
            link1.swap1 = True
        else:
            assert not link1.swap2
            link1.swap2 = True

        if link2.n1 == self:
            assert not link2.swap1
            link2.swap1 = True
        else:
            assert not link2.swap2
            link2.swap2 = True

        b = random.uniform(0, 1) <= self.topo.q

        if b:
            # Why don't we append a tuple with the links?
            self.internalLinks.append((link1, link2))
        return b

    # Function that creates an edge between two nodes
    def nodeTo(self, n2):
        return Edge(self.n1, n2)

    def __repr__(self):
        return f"Node id: {self.id}"
    # Need to figure out the string manipulation part

    # def toString(self):	pass

    # def toFullString(self):	pass



def to(node1, node2):
    return Edge(node1, node2)


"""This class defines an edge of the graph"""

class Edge:
    def __init__(self, n1, n2):
        self.p = (n1, n2)
        self.n1 = n1
        self.n2 = n2

    # Converts the tuple that conforms an edge into a list.
    def toList(self):
        return list(self.p)

    # Given a node n, returns the other node in the edge.
    def otherThan(self, n):
        if n == self.n1:
            return self.n2
        elif n == self.n2:
            return self.n1
        else:
            raise RuntimeError("Neither")

    # Returns true if the node n is either n1 or n2.
    def contains(self, n):
        return self.n1 == n or self.n2 == n

    # The hashcode of the edge is the xor function between the ids of both nodes.
    def hashCode(self):
        return self.n1.id ^ self.n2.id

    # An exact same edge shares both n1 and n2. Note that the edge is bidirectional.
    def equals(self, other):
        return (type(other) is Edge) and (self.p == other.p or reversed(self.p) == other.p)
    def __eq__(self, other):
        return self.equals(other)
    def __hash__(self):
        return hash((self.n1.id^self.n2.id))
