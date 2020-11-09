"""
Implementation of k-means
"""
import numpy as np
import sys
import networkx as nx
import matplotlib.pyplot as plt
from DistanceMetric import DistanceMetric
from sklearn import datasets
from Graphing import *

class KMeans_Naive():
    def __init__(self, graph, nClusters, distanceMetric="euclidean"):
        self.graph = graph
        self.nClusters = nClusters
        self.distanceMetric = DistanceMetric(distanceMetric)
        self.run()
    
    def run(self):
        range_max = self.graph.network.max(axis=0)
        range_min = self.graph.network.min(axis=0)
        # TODO Change this to select a point from among the points available
        # Give the user a choice to do this
        self.graph.clusterCenters = np.hstack([np.random.uniform(range_min[0],range_max[0],[self.nClusters,1]),np.random.uniform(range_min[1],range_max[1],[self.nClusters,1])])
        self.graph.show("Original/Initial points")
        self.graph.show("Original/Initial points + Initial cluster centers", True)
        oldCenters = newCenters = self.graph.clusterCenters
        i= 0
        while True :
            # get distance to all points
            oldCenters = newCenters.copy()
            distances = self.distanceMetric.distance(self.graph.clusterCenters, self.graph.network, self.nClusters)
            color = np.argmin(distances, axis=0)
            self.graph.color = color
            self.graph.clusterCenters = self.updateCenters(self.graph.clusterCenters)
            newCenters = self.graph.clusterCenters
            self.graph.show("Update: Iteration " + str(i), True)
            i += 1
            terminationCriteria = np.sum(self.distanceMetric.distance(oldCenters, newCenters, self.nClusters).diagonal()) < 0.1
            if terminationCriteria:
                self.graph.show("Final Clusters", True, True)
                break

    def updateCenters(self, clusterCenters):
        for i in range(self.nClusters):
            # Possible that no points are to be assigned to a cluster center
            # This is a problem. Restart and show that we are researting
            closePoints = self.graph.network[self.graph.color == i]
            if closePoints.size == 0:
                # Restart by calling the k-means' run function
                self.run()
                # Call system exit so that the previous failed attempts will stop
                sys.exit()
            else:
                clusterCenters[i] = np.mean(closePoints, axis=0)
        return clusterCenters

if __name__ == "__main__":
    numNodes = 1000
    circles, colors = datasets.make_moons(n_samples=1500, random_state=8, noise=0.05)
    points = MyPoints()
    points.setNetwork(circles, colors)
    kmeans = KMeans_Naive(points, 2)