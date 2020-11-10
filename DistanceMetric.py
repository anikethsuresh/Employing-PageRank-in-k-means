"""
Class to implement the distance metrics - Euclidean distance and Shortest distance to nodes in a Graph
"""

import time

import numpy as np
import networkx as nx

class DistanceMetric():
    """
    Class to perform distance calculation between nodes, points and vertices
    """
    def __init__(self, name="euclidean"):
        """
        Arguments:
        name - str, name of the distance metric
        """
        self.name = name
        # Distance between all pairs of nodes/points/vertices
        self.all_distances = None

    # def init_distances(self, Graph):
        # self.all_distances = dict(nx.all_pairs_shortest_path_length(Graph.graph))    


    def init_distances(self, Graph):
        """
        Initialize the distance matrix, which holds the distance between all pairs of nodes
        Implements the Floyd-Warshall algorithm for all pairs shortest paths
        """
        print("Calculating the distance between all pairs of nodes")
        start_time = time.time()
        nodes = list(Graph.nodes)
        edges = np.array(Graph.edges)
        # Set all distances to infinity
        distance = np.ones([len(nodes),len(nodes)]) * 9999
        # Set all self-distances to 0
        distance[nodes, nodes] = 0
        # Set the edges available to 1
        distance[edges[:,0], edges[:,1]] = distance[edges[:,1], edges[:,0]] = 1
        for k in range(len(nodes)):
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if distance[i][j] > distance[i][k] + distance[k][j]:
                        distance[i][j] =  distance[i][k] + distance[k][j]
        # Create dictionary where the keys are the nodes and the values are the distances to each node
        # This is the same format that networkx works.
        distances_dict = {}
        for i in range(len(nodes)):
            node_i_distances = distance[i]
            node_i_distances_dict = {x : node_i_distances[x] for x in range(len(nodes))}
            distances_dict[i] = node_i_distances_dict
        self.all_distances =  distances_dict
        stop_time = time.time()
        print("Completed.\nTook " + str(stop_time - start_time) + " sec")

    

    def distance(self,a ,b, nClusters):
        """
        Gets/Calculates the distance from a to b

        Arguments:
        a - nodes/clusters
        b - nodes/clusters
        nClusters - int, number of clusters
        """
        if self.name == "euclidean":
            # Distance metric for points in euclidean space
            # Calculates euclidean distance using numpy math
            nPoints = b.shape[0]
            b = np.transpose(b)
            aa1bb1 = -2 * np.dot(a,b)
            a2b2 = np.sum(pow(a,2),1)
            a12b12 = np.sum(pow(b,2),0)
            total = aa1bb1 + a12b12 + np.repeat(a2b2,nPoints).reshape([nClusters,nPoints])
            return np.sqrt(total)
        
        elif self.name == "floyd_warshall":
            # Gets the distance from the previous calculation
            # of the floyd warshall algorithm
            main_distances = {}
            distances = np.zeros([a.size, b.numNodes])
            index = 0
            for node in a:
                main_distances[node] = self.all_distances[node]
                for dist in main_distances[node].keys():
                    distances[index][dist] = main_distances[node][dist]
                index += 1
            return distances
    

    def distance3D(self, a, b, nClusters, actualVertices):
        """
        Calculates the distance from a to b for 3D points

        Arguments:
        a - nodes/clusters
        b - nodes/clusters
        nClusters - int, number of clusters
        actualVertices - dict, keys are the vertex as represented in trimesh, and the values are their actual vertex id
                        to ensure that no vertex is repeated
        """
        main_distances = {}
        for i in range(nClusters):
            a[i] = actualVertices[a[i]]
        distances = np.zeros([a.size, b.numNodes])
        index = 0
        for node in a:
            main_distances[node] = self.all_distances[node]
            for dist in main_distances[node].keys():
                distances[index][dist] = main_distances[node][dist]
            index += 1
        for x in range(distances.shape[0]):
            for y in range(distances.shape[1]):
                distances[x,y] = distances[actualVertices[x],actualVertices[y]]
        return distances


    def distance_between_centers(self, G, oldCenters, newCenters):
        """
        Calculates the distance between old cluster centroids and new cluster centroids

        Arguments:
        G - graph, networkx graph
        oldCenters - list, list of cluster centroids for previous iteration
        newCenters - list, list of cluster centroids for current iteration
        """
        distances = np.zeros([oldCenters.size])
        for i in range(oldCenters.size):
            new_distance = self.all_distances[oldCenters[i]].get(newCenters[i])
            if new_distance is None:
                distances[i] = 9999
            else:
                distances[i] = self.all_distances[oldCenters[i]].get(newCenters[i])
        return distances


    def distance_between_centers3D(self, G, oldCenters, newCenters, actualVertices):
        """
        Calculates the distance between old cluster centroids and new cluster centroids for 3D points

        Arguments:
        G - graph, networkx graph
        oldCenters - list, list of cluster centroids for previous iteration
        newCenters - list, list of cluster centroids for current iteration
        actualVertices - dict, keys are the vertex as represented in trimesh, and the values are their actual vertex id
                        to ensure that no vertex is repeated
        """
        distances = np.zeros([oldCenters.size])
        for i in range(oldCenters.size):
            distances[i] = self.all_distances[actualVertices[oldCenters[i]]].get(actualVertices[newCenters[i]])
            # Does not make sense to use PageRank on a disconnected graph
        return distances

if __name__ == "__main__":
    # Test to check that distance metric works successfully on 2d points
    test_DM = DistanceMetric()
    a = np.array([[1,2],[2,3],[3,4]])
    b = np.array([[3,6],[4,7],[5,8]])
    print(test_DM.distance(a,b,3))