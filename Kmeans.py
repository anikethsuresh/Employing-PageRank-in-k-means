"""
Implementation of k-means
"""
import numpy as np
import sys
class KMeans():
    def __init__(self, graph, nClusters, distanceMetric="euclidean"):
        self.graph = graph
        self.nClusters = nClusters
        self.distanceMetric = DistanceMetric(distanceMetric)
        self.run()
    
    def run(self):
        range_max = self.graph.network.max(axis=0)
        range_min = self.graph.network.min(axis=0)
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
            if np.sum(self.distanceMetric.distance(oldCenters, newCenters, self.nClusters).diagonal()) < 0.1:
                self.graph.show("Final Clusters", True, True)
                break

    def updateCenters(self, clusterCenters):
        for i in range(self.nClusters):
            # Possible that no points are to be assigned to a cluster center
            # This is a problem. Restart and show that we are researting
            closePoints = self.graph.network[self.graph.color == i]
            if closePoints.size == 0:
                # print("Restarting")
                # Restart by calling the k-means' run function
                self.run()
                # Call system exit so that the previous failed attempts will stop
                sys.exit()
            else:
                clusterCenters[i] = np.mean(closePoints, axis=0)
        return clusterCenters



class DistanceMetric():
    def __init__(self, name="euclidean"):
        self.name = name
    
    def distance(self,a ,b, nClusters):
        if self.name == "euclidean":
            nPoints = b.shape[0]
            b = np.transpose(b)
            aa1bb1 = -2 * np.dot(a,b)
            a2b2 = np.sum(pow(a,2),1)
            a12b12 = np.sum(pow(b,2),0)
            total = aa1bb1 + a12b12 + np.repeat(a2b2,nPoints).reshape([nClusters,nPoints])
            return np.sqrt(total)


if __name__ == "__main__":
    test_DM = DistanceMetric()
    a = np.array([[1,2],[2,3],[3,4]])
    b = np.array([[3,6],[4,7],[5,8]])
    print(test_DM.distance(a,b,3))