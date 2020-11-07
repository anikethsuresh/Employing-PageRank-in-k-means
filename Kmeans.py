"""
Implementation of k-means
"""
import numpy as np
import sys
import networkx as nx
import matplotlib.pyplot as plt
class KMeans():
    def __init__(self, graph, nClusters, distanceMetric="euclidean"):
        self.graph = graph
        self.nClusters = nClusters
        self.distanceMetric = DistanceMetric(distanceMetric)
        self.run()
    
    def run(self):
        range_max = self.graph.network.max(axis=0)
        range_min = self.graph.network.min(axis=0)
        # TODO Change this to select a point from amoong the points available
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

class KMeansWithPageRank():
    def __init__(self, myGraph, nClusters, distanceMetric="dijkstra"):
        self.graph = myGraph
        self.nClusters = nClusters
        self.distanceMetric = DistanceMetric("dijkstra")
        self.graph.init_adjacency_list(self.graph.edges)
        self.run()

    def run(self):
        # Select nCluster nodes as starting points
        self.graph.clusterCenters = np.random.randint(0, self.graph.numNodes, self.nClusters)
        # nx.draw_networkx_edges(self.graph.graph,self.graph.nodes,alpha=0.4)
        # nx.draw_networkx_nodes(self.graph.graph,self.graph.nodes,nodelist =list(self.graph.clusterCenters),node_color="#ff1c03",node_shape="X",
        #             node_size=80)
        # plt.pause(1)
        # plt.clf()
        for i in range(10):
        # Get distance of these clusters to all other nodes
            distances = self.distanceMetric.distance(self.graph.clusterCenters, self.graph, self.nClusters)
            self.colors = np.argmin(distances, axis=0)
            nx.draw_networkx_edges(self.graph.graph,self.graph.nodes,alpha=0.4)
            nx.draw_networkx_nodes(self.graph.graph,self.graph.nodes, node_color=list(self.colors),cmap=plt.cm.jet,
                        node_size=20)
            nx.draw_networkx_nodes(self.graph.graph,self.graph.nodes,nodelist =list(self.graph.clusterCenters),node_color="#ff1c03",node_shape="X",
                        node_size=80)
            labels = {x:str(x) for x in self.graph.clusterCenters}
            nx.draw_networkx_labels(self.graph.graph,self.graph.nodes, labels=labels, font_color="#02CC16")
            plt.title("Iteration:" + str(i + 1))
            plt.pause(1)
            plt.clf()
            # Get PageRank for each cluster
            self.updateCenters()

    def updateCenters(self):
        for cluster_i in range(self.nClusters):
            nodes = np.where(self.colors == cluster_i)[0]
            adjacency_matrix_cluster_i = np.zeros([nodes.size, nodes.size])
            self.graph.fill_adjacency_list(adjacency_matrix_cluster_i, nodes)
            cluster_page_rank = self.graph.page_rank(adjacency_matrix_cluster_i, len(nodes))
            self.graph.clusterCenters[cluster_i] =  nodes[np.argmax(cluster_page_rank)]
            

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
            


if __name__ == "__main__":
    test_DM = DistanceMetric()
    a = np.array([[1,2],[2,3],[3,4]])
    b = np.array([[3,6],[4,7],[5,8]])
    print(test_DM.distance(a,b,3))