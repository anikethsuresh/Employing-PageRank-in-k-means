import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from DistanceMetric import DistanceMetric
from sklearn import datasets
from Graphing import MyGraph
from KMeans_Naive import KMeans_Naive
from KMeansWithPageRank import KMeansWithPageRank

def get_edges(points, threshold):
    """
    Keyword arguments:
    nodes: dictionary with the key as the node id and value as the x,y point in 2D euclidean space
    """
    distanceMetric = DistanceMetric("euclidean")
    dist = distanceMetric.distance(points, points, len(points))
    dist = np.nan_to_num(dist)
    edges = []
    for x in range(dist.shape[0]):
        for y in range(x+1,dist.shape[1]):
            if dist[x,y] < threshold:
                edges.append((x,y))
    return edges

if __name__ == "__main__":
    numNodes = 1000
    circles, _ = datasets.make_moons(n_samples=numNodes, noise=0.05)
    # Get all nodes in the format that networkx needs
    nodes = {}
    for point in range(len(circles)):
        nodes[point] = tuple(circles[point])
    edges = get_edges(circles, 0.1)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    myGraph = MyGraph(G, numNodes, nodes, edges)
    nx.draw_networkx_edges(G,nodes,alpha=0.4, edge_color="#424242")
    nx.draw_networkx_nodes(G,nodes,node_size=30,nodelist=list(range(len(circles))),node_color='r', alpha = 0.5)
    plt.show()
    KMeansWithPageRank(myGraph, 2)