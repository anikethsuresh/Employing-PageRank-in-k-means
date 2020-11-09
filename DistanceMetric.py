import numpy as np
import networkx as nx

class DistanceMetric():
    def __init__(self, name="euclidean"):
        self.name = name
        self.all_distances = None
    
    def init_distances(self, network):
        self.all_distances = dict(nx.all_pairs_shortest_path_length(network))

    def distance(self,a ,b, nClusters):
        if self.name == "euclidean":
            nPoints = b.shape[0]
            b = np.transpose(b)
            aa1bb1 = -2 * np.dot(a,b)
            a2b2 = np.sum(pow(a,2),1)
            a12b12 = np.sum(pow(b,2),0)
            total = aa1bb1 + a12b12 + np.repeat(a2b2,nPoints).reshape([nClusters,nPoints])
            return np.sqrt(total)
        
        elif self.name == "dijkstra":
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
        distances = np.zeros([oldCenters.size])
        for i in range(oldCenters.size):
            new_distance = self.all_distances[oldCenters[i]].get(newCenters[i])
            if new_distance is None:
                distances[i] = 9999
            else:
                distances[i] = self.all_distances[oldCenters[i]].get(newCenters[i])
        return distances

    def distance_between_centers3D(self, G, oldCenters, newCenters, actualVertices):
        distances = np.zeros([oldCenters.size])
        for i in range(oldCenters.size):
            distances[i] = self.all_distances[actualVertices[oldCenters[i]]].get(actualVertices[newCenters[i]])
            # Does not make sense to use PageRank on a disconnected graph
        return distances

if __name__ == "__main__":
    test_DM = DistanceMetric()
    a = np.array([[1,2],[2,3],[3,4]])
    b = np.array([[3,6],[4,7],[5,8]])
    print(test_DM.distance(a,b,3))