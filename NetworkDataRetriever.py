import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from Kmeans import KMeansWithPageRank
from Graphing import MyGraph

if __name__ == "__main__":
    numNodes = 500
    G = nx.random_geometric_graph(numNodes,0.16, seed=1643)
    nodes = nx.get_node_attributes(G,'pos')
    edges = list(G.edges)
    myGraph = MyGraph(G, numNodes, nodes, edges)
    KMeansWithPageRank(myGraph, 4)
    # myGraph.init_adjacency_list(edges)
    # colors = myGraph.page_rank(myGraph.adjacency_list, numNodes)
    # nx.draw_networkx_edges(G,nodes,alpha=0.4)
    # nx.draw_networkx_nodes(G,nodes, node_color=list(colors),cmap=plt.cm.jet,
    #                    node_size=20)
    # plt.show()

