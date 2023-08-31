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
max_len = 10  # maximum path length # TODO: way to calculate min/max len
n_demands = 8  # Total number of demands
q = 0.5  # BSM success probability
repeat = 1
rand_seed = 85
node_count = 30

Arcs = []   #links
D = []  # Modified demand set
#length_path = []  # path length constraint,  deleted hardcoded length
D_acc = []  # Actual demand set

Aff = []  # Variables for the LP objective function
vars = []  # LP variables






def simpleTest():
    global topo
    netTopology = Topo.generateString(node_count, 0.6, 5, 0.1, 6)

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
nodes, edges, edge_length = format_topology(topo)

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


G.nodedic = {} # dictionary of node to modified nodes

#print("After nx:", G.nodes) # e.g. [1,...,30,0]

# Add the edges to the graph
# Since the edges include weights, we can use the add_weighted_edges_from function
weighted_edges = [(a+1, b+1, weight_dict['weight']) for a, b, weight_dict in edges]
#weighted_edges = [(a+1, b+1, weight_dict['weight'], edge_length[(a, b)]) for a, b, weight_dict in edges]
#weighted_edges = [(a+1, b+1, weight_dict['weight'], edge_length.get((a, b), 0)) for a, b, weight_dict in edges]


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

print("Dictionary:", G.nodedic)
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
D_acc = [(a+1, b+1) for a, b in D_acc] # prevent negative node/demand
l = randint(min_len, max_len)

for x in D_acc:
    for k in range(l):
        D.append(((x[0] - 1) * (max_len + 1) + 1, (x[1] - 1) * (max_len + 1) + k + 2))
        a ,b = x[0], x[1]
        #length_path.append(k+1)

# output
print("Demands: ", D_acc)
print("Modified Demand: ", D)
print("Link: ", topo.links)
print("Link length:", edge_length)
#print("Lengths: ", length_path)
tot_dem = len(D)

sum_in = [None] * len(Nodes_mod) * tot_dem
sum_out = [None] * len(Nodes_mod) * tot_dem

# Creates the boundless Variables as real numbers
for k in range(tot_dem): # for each demand
    temp_vars = []

    for (i, j, w) in Arcs_mod: # for each src, dst, weight pair
        #x = LpVariable(str((i, j, k)), lowBound=0, upBound=w, cat='Continuous')
        #print(f"{i},{j},{k}")
        x = LpVariable(f"({i},{j},{k})", lowBound=0, upBound=w, cat='Continuous')
        temp_vars.append((i, j, k, x))

        if (i == D[k][0]):  # if src = Demand's src
            Aff.append((x, q ** (edge_length[G.nodedic[i] - 1, G.nodedic[j] - 1])))
            #Aff.append((x, q ** (length_path[k] - 1)))
            #print(G.nodedic[i], G.nodedic[j])
            #print(edge_length[G.nodedic[i]-1, G.nodedic[j]-1])


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
    if v.varValue > 1e-10:
        print(v.name, "=", v.varValue)
        non_zero_var.append(v)

result_list = [[] for _ in range(len(D))]
for v in prob.variables():# TODO: convert back to actual nodes
    if v.varValue > 1e-10:
        a, b, c = [int(num) for num in v.name.strip("()").split(",")]  # convert string in to num
        #print(a,b,c)
        result_list[c].append([G.nodedic[a],G.nodedic[b],v.varValue])
        #print(G.nodedic[a], G.nodedic[b], c, "=", v.varValue)  # get actual nodes
        non_zero_var.append(v)
#print(result_list)
my_dict = {k: 0 for k in D_acc}  # total path count for each demand

for ren, result in enumerate(result_list):
    key_tuple = (G.nodedic[D[ren][0]], G.nodedic[D[ren][1]])  # key for dictionary
    for connection in result:  # add to path count
        if connection[0] == G.nodedic[D[ren][0]]:
            my_dict[key_tuple] += connection[2]
            print(connection[0], G.nodedic[D[ren][0]], my_dict[key_tuple], connection[2])
    print(ren, result, G.nodedic[D[ren][0]], G.nodedic[D[ren][1]])


# The optimised objective function value is printed to the screen
# (Modified_Node1, Modified_Node2, # of Modified_Demand) = # of Links
print("Total Achievable Rate = ", value(prob.objective))

#for i, demand in enumerate(D_acc):
    #print(str(demand[0]) + "<->" + str(demand[1]), "=", my_dict[D_acc[i]], end="  ")
print("\n\n\n\n\nFINAL OUTPUT:\n")
output = '  '.join(f"{key[0]}<->{key[1]} × {value}" for key, value in my_dict.items())
print(output)
#print(my_dict)
