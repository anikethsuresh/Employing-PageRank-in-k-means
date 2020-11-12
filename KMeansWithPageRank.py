"""
Definition of the KMeansWithPageRank class, which perform k-means using PageRank
"""

import sys

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from DistanceMetric import DistanceMetric
from Graphing import *


class KMeansWithPageRank():
    """
    Provides the capability to run k-means with PageRank as the update mechanism
    """
    def __init__(self, myGraph, nClusters, distanceMetric="floyd_warshall", showEdges=True):
        """
        Arguments:
        myGraph - MyGraph object
        nClusters - int, number of clusters
        distanceMetric - distance metric to be used. Could either be euclidean for points in euclidean space,
                        or shortest distance in a graph. Dijkstra's isn't 
        showEdges - boolean, determines whether the edges should be shown in visualization
        """
        self.graph = myGraph
        self.nClusters = nClusters
        self.distanceMetric = DistanceMetric("floyd_warshall")
        self.showEdges = showEdges
        self.distanceMetric.init_distances(myGraph)
        self.run()

    def run(self):
        """
        Runs the k-means algorithm with PageRank
        """

        # Initialize the clusters as one among the points in the graph
        self.graph.clusterCenters = np.random.randint(0, self.graph.numNodes, self.nClusters)
        i = 0
        while True:
            oldClusterCenters = self.graph.clusterCenters.copy()
            # Distance, visualization and termination criteria differ based on the type of graph
            if isinstance(self.graph, My3DGraph):
                # Get the distance from the clusters to all the points in the graph
                distances = self.distanceMetric.distance3D(self.graph.clusterCenters, self.graph, self.nClusters, self.graph.actualVertices)
                # Set the points' cluster, to the cluster center it is closest to
                self.colors = np.argmin(distances, axis=0)
                self.graph.show( self.colors)
                # Update the new clusters
                self.updateCenters()
                newClusterCenters = self.graph.clusterCenters
                # Stop the program if the cluster centers do not move (much).
                terminationCriteria = self.distanceMetric.distance_between_centers3D(self.graph.graph, oldClusterCenters, newClusterCenters, self.graph.actualVertices)
                # If the clusters did not shift in between iterations, we have reached local maximum
                terminationCriteria = terminationCriteria[terminationCriteria==0].size/self.graph.clusterCenters.size > 0.5 and np.all(oldClusterCenters == newClusterCenters)
            elif isinstance(self.graph, MyGraph):
                # Get the distance from the clusters to all the points in the graph
                distances = self.distanceMetric.distance(self.graph.clusterCenters, self.graph, self.nClusters)
                # Set the points' cluster, to the cluster center it is closest to
                self.colors = np.argmin(distances, axis=0)
                self.graph.show("Iteration:" + str(i + 1), self.colors, withCenters=True, showEdges=self.showEdges)
                self.updateCenters()
                newClusterCenters = self.graph.clusterCenters
                terminationCriteria = self.distanceMetric.distance_between_centers(self.graph.graph, oldClusterCenters, newClusterCenters)
                # Stop the program if the two clusters are disconnected in the graph, or if they do not move
                terminationCriteria = np.all(terminationCriteria == terminationCriteria[0])
            i += 1
            if terminationCriteria:
                self.graph.clusterCenters = oldClusterCenters
                if isinstance(self.graph, My3DGraph):
                    print("THE END")
                    self.graph.show(self.colors)
                elif isinstance(self.graph, MyGraph):
                    self.graph.show("Final Clusters (Total Iterations: " + str(i) + ")", self.colors, final=True, showEdges=self.showEdges)
                break
            

    def updateCenters(self):
        """
        Updates the cluster center to move towards the points which have the highest PageRank value (from the PageRank vector)
        """
        for cluster_i in range(self.nClusters):
            nodes = np.where(self.colors == cluster_i)[0]
            if nodes.size == 0:
                self.run()
                sys.exit()
            # Extract the points belonging to a cluster
            adjacency_matrix_cluster_i = np.zeros([nodes.size, nodes.size])
            # Set the adjacency list to represent the points in this cluster
            self.graph.fill_adjacency_list(adjacency_matrix_cluster_i, nodes)
            # Get the PageRank vector for the points in this cluster
            cluster_page_rank = self.graph.page_rank(adjacency_matrix_cluster_i, len(nodes))
            # Set the new cluster center as the point with the largest pagerank
            self.graph.clusterCenters[cluster_i] =  nodes[np.argmax(cluster_page_rank)]

if __name__ == "__main__":
    numNodes = 500
    G = nx.random_geometric_graph(numNodes,0.16, seed=1643)
    nodes = nx.get_node_attributes(G,'pos')
    edges = list(G.edges)
    myGraph = MyGraph(G, numNodes, nodes, edges)
    myGraph.init_adjacency_list(edges)
    colors = myGraph.page_rank(myGraph.adjacency_list, numNodes)
    myGraph.show("PageRank over the entire graph", colors)

    KMeansWithPageRank(myGraph, 4)