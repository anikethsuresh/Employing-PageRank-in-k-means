import numpy as np
import networkx as nx

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
        
        elif self.name == "dijkstra":
            all_distances = dict(nx.all_pairs_shortest_path_length(b.graph))
            main_distances = {}
            distances = np.zeros([a.size, b.numNodes])
            index = 0
            for node in a:
                main_distances[node] = all_distances[node]
                for dist in main_distances[node].keys():
                    distances[index][dist] = main_distances[node][dist]
                index += 1
            return distances

    def distance_between_centers(self, oldCenters, newCenters, adjacency_list):
        distances = np.zeros([oldCenters.size])
        for i in range(oldCenters.size):
            distances[i] = adjacency_list[oldCenters[i]][newCenters[i]]
        return distances

if __name__ == "__main__":
    test_DM = DistanceMetric()
    a = np.array([[1,2],[2,3],[3,4]])
    b = np.array([[3,6],[4,7],[5,8]])
    print(test_DM.distance(a,b,3))