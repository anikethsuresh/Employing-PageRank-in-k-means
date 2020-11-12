"""
The following code implements the naive k-means algorithm
"""

import sys

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import datasets

from DistanceMetric import DistanceMetric
from Graphing import *

class KMeans_Naive():
    """
    Implementation of the basic k-means algorithm (Lloyd's algorithm)
    """
    def __init__(self, graph, nClusters, distanceMetric="euclidean"):
        """
        Arguments:
        graph - networkx graph
        nClusters - int, number of clusters
        distanceMetric - str, euclidean distance is used for points in Euclidean space
        """
        self.graph = graph
        self.nClusters = nClusters
        self.distanceMetric = DistanceMetric(distanceMetric)
        self.run()
    

    def run(self):
        """
        Runs the k-means algorithm
        """
        range_max = self.graph.network.max(axis=0)
        range_min = self.graph.network.min(axis=0)
        # Initialize the cluster center in the space that the points occur
        self.graph.clusterCenters = np.hstack([np.random.uniform(range_min[0],range_max[0],[self.nClusters,1]),np.random.uniform(range_min[1],range_max[1],[self.nClusters,1])])
        self.graph.show("Original/Initial points")
        self.graph.show("Original/Initial points + Initial cluster centers", True)
        oldCenters = newCenters = self.graph.clusterCenters
        i= 0
        while True :
            oldCenters = newCenters.copy()
            # Get the distance from the clusters to all the points in the graph
            distances = self.distanceMetric.distance(self.graph.clusterCenters, self.graph.network, self.nClusters)
            # Set the points' cluster, to the cluster center it is closest to
            color = np.argmin(distances, axis=0)
            self.graph.color = color
            # Update the new clusters
            self.graph.clusterCenters = self.updateCenters(self.graph.clusterCenters)
            newCenters = self.graph.clusterCenters
            self.graph.show("Update: Iteration " + str(i), True)
            i += 1
            terminationCriteria = np.sum(self.distanceMetric.distance(oldCenters, newCenters, self.nClusters).diagonal()) < 0.1
            # Stop the program if the cluster centers do not move (much). Threshold for movement in euclidean distance is set to 0.1
            if terminationCriteria:
                self.graph.show("Final Clusters", True, True)
                break

    def updateCenters(self, clusterCenters):
        """
        Updates the cluster centers to the mean of the points belonging to the cluster

        Arguments:
        clusterCenters - coordinates of the center of the clusters
        """
        for i in range(self.nClusters):
            # Possible that no points are to be assigned to a cluster center
            # This is possible if the points move away from the centroids, or if two cluster centers
            # are too close to each other. Restart the program
            closePoints = self.graph.network[self.graph.color == i]
            if closePoints.size == 0:
                # Restart by calling the k-means' run function
                self.run()
                # Call system exit so that the previous failed attempts will stop
                sys.exit()
            else:
                # Update the cluster point to the mean of the points beloning to this cluster
                clusterCenters[i] = np.mean(closePoints, axis=0)
        return clusterCenters

if __name__ == "__main__":
    numNodes = 1000
    circles, colors = datasets.make_blobs(n_samples=numNodes, random_state=8)
    points = MyPoints()
    points.setNetwork(circles, colors)
    kmeans = KMeans_Naive(points, 2)