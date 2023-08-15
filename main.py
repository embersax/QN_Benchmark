import matplotlib

matplotlib.use('TkAgg')
import networkx as nx
from pulp import *
from random import seed
from random import randint
from random import shuffle
from Grap_reading import Mod_net
from math import exp
from  topo.Topo import *
from utils.utils import *
import warnings
warnings.filterwarnings("ignore")


min_len = 1  # minimum path length
max_len = 10  # maximum path length
n_demands = 5  # Total number of demands
q = 0.5  # BSM success probability
repeat = 1
rand_seed = 85

Arcs = []   #links
D = []  # Modified demand set
length_path = []  # path length constraint
D_acc = []  # Actual demand set

Aff = []  # Variables for the LP objective function
vars = []  # LP variables






def simpleTest():
    global topo
    netTopology = Topo.generateString(30, 0.6, 5, 0.1, 6)

    alphas = []
    p = [.8, .5]
    for expectedAvgP in p:
        alpha = step = .1
        lastAdd = True
        while True:
            lines = list(netTopology.split('\n'))
            lines[1] = str(alpha)
            topo = Topo('\n'.join(lines))

            avgP = float(sum(list(map(lambda it: exp(-alpha * length([it.n1.loc, it.n2.loc])), topo.links)))) / len(
                topo.links)

            if abs(avgP - expectedAvgP) / expectedAvgP < .001:
                break

            elif avgP > expectedAvgP:
                if not lastAdd: step /= 2
                alpha += step
                lastAdd = True

            else:
                if lastAdd: step /= 2
                alpha -= step
                lastAdd = False

        alphas.append(alpha)
    return topo

topo = simpleTest()

# Get the nodes and edges from format_topology
nodes, edges = format_topology(topo)

#print("Original Nodes:") # e.g. [0,1,...,29]
#print(nodes)

# Increment each node by 1, prevent modified nodes to be negative
nodes = [node + 1 for node in nodes]

#print("Original Nodes:") # e.g. [1,...,30]
#print(nodes)


# Create a new graph
G = nx.Graph()

# Add the nodes to the graph
G.add_nodes_from(nodes)
#print("After nx:", G.nodes) # e.g. [1,...,30,0]

# Add the edges to the graph
# Since the edges include weights, we can use the add_weighted_edges_from function
weighted_edges = [(a, b, weight_dict['weight']) for a, b, weight_dict in edges]

G.add_weighted_edges_from(weighted_edges)
print(G.nodes)
print(G.edges(data=True))
n = len(list(G.nodes()))
# Seed for generating random numbers
seed(rand_seed)

for (i, j) in G.edges():
    Arcs.append((i, j, list(G.edges[i, j].values())[0]))

(Nodes_mod, Arcs_mod) = Mod_net(G, max_len)  # Create the modified network
N = len(Nodes_mod)  # Total number of nodes in the modified network
G_mod = nx.DiGraph()  # Modified network
G_mod.add_nodes_from(Nodes_mod)
G_mod.add_weighted_edges_from(Arcs_mod)


# Demand Creation
def funInLine135(combs, nsd):
    shuffle(combs)
    return list(map(lambda it: (it[0].id, it[1].id), combs[:nsd]))


testSetIter, testSet = [i for i in range(n_demands, n_demands+1)], []
for nsd in testSetIter:
    combs = list(combinations(topo.nodes, 2))
    testSet.append(funInLine135(combs, nsd))#, repeat))
#print("testset:", testSet)  # e.g. testSet = [[(12, 13)], [(4, 27), (1, 8)]]

D_acc = [pair for sublist in testSet for pair in sublist] # e.g. Demands:  [(12, 13), (4, 27), (1, 8)]

l = randint(min_len, max_len)

for x in D_acc:
    for k in range(l):
        D.append(((x[0] - 1) * (max_len + 1) + 1, (x[1] - 1) * (max_len + 1) + k + 2))
        a ,b = x[0], x[1]
        length_path.append(k+1) # refering to
                                  # linkLengths = sorted([(link.node1.loc - link.node2.loc) for link in self.links])


# output
print("Demands: ", D_acc)
print("Modified Demand: ", D)
print("Lengths: ", length_path)
tot_dem = len(D)

sum_in = [None] * len(Nodes_mod) * tot_dem
sum_out = [None] * len(Nodes_mod) * tot_dem

# Creates the boundless Variables as real numbers
for k in range(tot_dem):
    temp_vars = []

    for (i, j, w) in Arcs_mod:
        #x = LpVariable(str((i, j, k)), lowBound=0, upBound=w, cat='Continuous')
        #print(f"{i},{j},{k}")
        x = LpVariable(f"({i},{j},{k})", lowBound=0, upBound=w, cat='Continuous')
        temp_vars.append((i, j, k, x))

        if (i == D[k][0]):
            Aff.append((x, q ** (length_path[k] - 1)))

    vars.append(temp_vars)

# Creates the 'prob' variable to contain the problem data
prob = LpProblem("Routing Problem", LpMaximize)

# Creates the objective function

flow = LpAffineExpression(Aff)
print("flow:", flow)
prob += flow, "Total Rate"

# Creates all problem constraints - this ensures the amount going into each node is at least equal to
# the amount leaving

for k in range(tot_dem):
    s = D[k][0]
    t = D[k][1]
    for v in Nodes_mod:

        if (v != s and v != t):
            sum_out[k * N + v - 1] = lpSum([x] for (i, j, l, x) in vars[k] if i == v and i != t and j != s)
            sum_in[k * N + v - 1] = lpSum([x] for (i, j, l, x) in vars[k] if j == v and i != t and j != s)

# flow conservation
for k in range(tot_dem):
    s = D[k][0]
    t = D[k][1]
    for v in G_mod.nodes:
        if (v != s and v != t):
            prob += sum_in[k * N + v - 1] == sum_out[k * N + v - 1]

# capacity constraints
Arcs_undir = []
for (u, v, w) in Arcs:
    if (u, v, w) not in Arcs_undir and (v, u, w) not in Arcs_undir:
        Arcs_undir.append((u, v, w))
edge_cap = [None] * len(Arcs)
for (u, v, w) in Arcs_undir:
    temp_var = []
    for k in range(tot_dem):
        for m in range(max_len + 1):
            for (i, j, l, x) in vars[k]:
                if ((i == (u - 1) * (max_len + 1) + m) and j == ((v - 1) * (max_len + 1) + m + 1)) or (
                        i == ((v - 1) * (max_len + 1) + m) and j == (u - 1) * (max_len + 1) + m + 1):
                    temp_var.append(x)
    prob += lpSum([x] for x in temp_var) <= w

# The problem is solved using PuLP's choice of Solver
prob.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

# Each of the variables is printed with it's resolved optimum value
non_zero_var = []
for v in prob.variables():
    if v.varValue > 0:
        print(v.name, "=", v.varValue)
        non_zero_var.append(v)

# The optimised objective function value is printed to the screen
# (Modified_Node1, Modified_Node2, # of Modified_Demand) = # of Links
print("Total Achievable Rate = ", value(prob.objective))
