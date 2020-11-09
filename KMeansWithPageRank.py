from DistanceMetric import DistanceMetric
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from Graphing import *
import sys

class KMeansWithPageRank():
    def __init__(self, myGraph, nClusters, distanceMetric="dijkstra", showEdges=True):
        self.graph = myGraph
        self.nClusters = nClusters
        self.distanceMetric = DistanceMetric("dijkstra")
        self.showEdges = showEdges
        self.distanceMetric.init_distances(myGraph.graph)
        self.run()

    def run(self):
        self.graph.clusterCenters = np.random.randint(0, self.graph.numNodes, self.nClusters)
        i = 0
        while True:
            oldClusterCenters = self.graph.clusterCenters.copy()
            if isinstance(self.graph, My3DGraph):
                distances = self.distanceMetric.distance3D(self.graph.clusterCenters, self.graph, self.nClusters, self.graph.actualVertices)
                self.colors = np.argmin(distances, axis=0)
                self.graph.show("Iteration:" + str(i + 1), self.colors)
                self.updateCenters()
                newClusterCenters = self.graph.clusterCenters
                terminationCriteria = self.distanceMetric.distance_between_centers3D(self.graph.graph, oldClusterCenters, newClusterCenters, self.graph.actualVertices)
                print(terminationCriteria)
                terminationCriteria = terminationCriteria[terminationCriteria==0].size/self.graph.clusterCenters.size > 0.5 and np.all(oldClusterCenters == newClusterCenters)
            elif isinstance(self.graph, MyGraph):
                distances = self.distanceMetric.distance(self.graph.clusterCenters, self.graph, self.nClusters)
                self.colors = np.argmin(distances, axis=0)
                self.graph.show("Iteration:" + str(i + 1), self.colors, withCenters=True, showEdges=self.showEdges)
                self.updateCenters()
                newClusterCenters = self.graph.clusterCenters
                terminationCriteria = self.distanceMetric.distance_between_centers(self.graph.graph, oldClusterCenters, newClusterCenters)
                # If they are disconnected the next statement will be true
                terminationCriteria = np.all(terminationCriteria == terminationCriteria[0])
            i += 1
            if terminationCriteria:
                self.graph.clusterCenters = oldClusterCenters
                if isinstance(self.graph, My3DGraph):
                    self.graph.show("Final", self.colors)
                elif isinstance(self.graph, MyGraph):
                    self.graph.show("Final Clusters (Total Iterations: " + str(i) + ")", self.colors, withCenters=True, final=True, showEdges=self.showEdges)
                break
            

    def updateCenters(self):
        for cluster_i in range(self.nClusters):
            nodes = np.where(self.colors == cluster_i)[0]
            if nodes.size == 0:
                self.run()
                sys.exit()
            adjacency_matrix_cluster_i = np.zeros([nodes.size, nodes.size])
            # TODO Optimize the code below
            self.graph.fill_adjacency_list(adjacency_matrix_cluster_i, nodes)
            cluster_page_rank = self.graph.page_rank(adjacency_matrix_cluster_i, len(nodes))
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

    KMeansWithPageRank(myGraph, 6)