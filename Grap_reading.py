
import matplotlib.pyplot as plt     #creating static, animated, and interactive visualizations in Python
import plotly.graph_objects as go   #creating rich interactive visualizations
import networkx as nx               #for the creation, manipulation, and study of complex networks.
import scipy as sp                  # for scientific and technical computing.
from random import seed
from random import randint



def graph_read(S):

    """Read the input graph from a .graphml.xml file. Here we consider the 'Surfnet.graphml.xml' as the input file."""
    H = nx.read_graphml(S)

    max=400
    Nodes = list(H.nodes)
    #print(len(Nodes))
    Arcs = []
    Wt = []
    G=nx.DiGraph()

    for i in Nodes:
        G.add_node(int(i)+1)

    i=0
    seed(1)
    for (n,m,w) in H.edges:
        x= randint(1,max)
        Arcs.append((int(n)+1,int(m)+1,x))
        Arcs.append((int(m)+1, int(n)+1, x))

    G.add_weighted_edges_from(Arcs)

    # New code to print the graph as text:
    print("Nodes of graph:")
    print(G.nodes())
    print("\nEdges of graph:")
    print(G.edges(data=True))
    return G




def Mod_net(G,l):

    """Modify the original network G for applying the length constrained multi-commodity flow formulation.
For each node u in the original graph G = (V,E,C) we create l+1 copies of u. They are u+0, ... , u+l.
If two nodes u,v in the original network G is connected by an edge (u,v), then in the modified network G'=(V',E',C')
we will have the following edges,
(u+0,v+1), (u+1,v+2), ... (u+(l-1),v+l)."""

    Nodes = []
    Arcs = []
    #print("G.nodes:", G.nodes)     # strange bug which G.nodes will have append a node "0" at the end
    for i in G.nodes:
        #print(i)
        for j in range(l+1):
            #print(i) #print the nodes
            Nodes.append((i-1)*(l+1)+j+1)
            #print("Node", i,j,(i-1)*(l+1)+j+1)
    for (i,j) in G.edges():
        for k in range(l):
            #Arcs.append(((i-1) * (l + 1) + k + 1, (j-1) * (l + 1) + k + 2, G[i][j]['weight']))
            Arcs.append(((i-1)*(l+1)+k+1,(j-1)*(l+1)+k+2,list(G.edges[i,j].values())[0]))

    print("Nodes of modified graph:")
    print(Nodes)
    print("\nArcs of modified graph:")
    print(Arcs)
    return (Nodes,Arcs)




