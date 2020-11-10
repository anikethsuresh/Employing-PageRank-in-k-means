"""
Implementation of the k-means algorithm on points in euclidean space
Rather than using the mean as the metric, we use PageRank
This ensure that clusters that would otherwise not converge to an optimum,
would now do so
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import datasets

from DistanceMetric import DistanceMetric
from Graphing import MyGraph
from KMeansWithPageRank import KMeansWithPageRank

def get_edges(points, threshold):
    """
    Keyword arguments:
    points: array of x,y (2D) points
    threshold: distance, used to determine whether two points are neighbours
    """
    distanceMetric = DistanceMetric("euclidean")
    dist = distanceMetric.distance(points, points, len(points))
    dist = np.nan_to_num(dist)
    edges = []
    # For all the points, check whether the distance between them are less than the threshold
    # If the distance is less than the threshold, add an edge between these nodes
    for x in range(dist.shape[0]):
        for y in range(x+1,dist.shape[1]):
            if dist[x,y] < threshold:
                edges.append((x,y))
    return edges

if __name__ == "__main__":
    numNodes = 500
    circles, _ = datasets.make_moons(n_samples=numNodes, random_state=8, noise=0.05)
    nodes = {}
    for point in range(len(circles)):
        nodes[point] = tuple(circles[point])
    edges = get_edges(circles, 0.15)
    # Initialize networkx graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    # Initialize MyGraph
    myGraph = MyGraph(G, numNodes, nodes, edges)
    # Get the pagerank vector for all the points. Stronger reds indicate important nodes.
    colors = myGraph.page_rank(myGraph.adjacency_list, numNodes)
    myGraph.show("Coloured based on PageRank vector", colors, showEdges=False)
    KMeansWithPageRank(myGraph, 2, showEdges=False)