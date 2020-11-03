"""
Implementation of k-means
"""
import numpy as np
class KMeans():
    def __init__(self, graph, clusters, distanceMetric="euclidean"):
        self.graph = graph
        self.clusters = clusters
        self.distanceMetric = DistanceMetric(distanceMetric)
        self.run()
    
    def run(self):
        range_max = self.graph.network.max(axis=0)
        range_min = self.graph.network.min(axis=0)
        clusterCenters = np.hstack([np.random.uniform(range_min[0],range_max[0],[self.clusters,1]),np.random.uniform(range_min[1],range_max[1],[self.clusters,1])])
        self.graph.show()
        for i in range(2):
            # get distance to all points
            distances = self.distanceMetric.distance(clusterCenters, self.graph.network, self.clusters)
            colors = np.argmin(distances, axis=1)
            self.graph.colors = colors
            self.graph.show()




class DistanceMetric():
    def __init__(self, name="euclidean"):
        self.name = name
    
    def distance(self,a ,b, clusters):
        if self.name == "euclidean":
            nPoints = b.shape[0]
            b = np.transpose(b)
            aa1bb1 = -2 * np.dot(a,b)
            a2b2 = np.sum(pow(a,2),1)
            a12b12 = np.sum(pow(b,2),0)
            total = aa1bb1 + np.transpose(np.repeat(a12b12,clusters).reshape([clusters,nPoints])) + np.repeat(a2b2,clusters).reshape([clusters,nPoints])
            return np.sqrt(total)


if __name__ == "__main__":
    test_DM = DistanceMetric()
    a = np.array([[1,2],[2,3],[3,4]])
    b = np.array([[3,6],[4,7],[5,8]])
    print(test_DM.distance(a,b,3))