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
    circles, _ = datasets.make_moons(n_samples=numNodes, random_state=8, noise=0.05)
    nodes = {}
    for point in range(len(circles)):
        nodes[point] = tuple(circles[point])
    edges = get_edges(circles, 0.15)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    myGraph = MyGraph(G, numNodes, nodes, edges)
    colors = myGraph.page_rank(myGraph.adjacency_list, numNodes)
    myGraph.show("Neighbourhood Graph", colors, showEdges=False)
    KMeansWithPageRank(myGraph, 2, showEdges=False)