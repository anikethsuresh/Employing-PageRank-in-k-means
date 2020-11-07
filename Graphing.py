"""
Graphing library
"""
import numpy as np
import matplotlib.pyplot as plt

class Points():
    def __init__(self):
        self.network = np.array([])
        self.color = np.array([])
        self.clusterCenters = np.array([])

    def addPoint(self, point, color):
        self.network = np.vstack((self.network,point))
        self.color = np.vstack((self.color,color))
    
    def setNetwork(self, network, color):
        self.network = network
        self.color = color
    
    def show(self, title, withCenters = False, final=False):
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.scatter(self.network[:,0],self.network[:,1], c=self.color, alpha=0.3, edgecolors="face", zorder=0)
        simpleColor= np.arange(0,self.clusterCenters.shape[0])
        if withCenters:
            plt.scatter(self.clusterCenters[:,0],self.clusterCenters[:,1], marker="x", zorder=10)
        plt.pause(1)
        if final:
            plt.show()
        else:
            plt.clf()

class MyGraph():
    def __init__(self, networkxGraph, numNodes, nodes, edges):
        self.graph = networkxGraph
        self.numNodes = numNodes
        self.adjacency_list = np.zeros([numNodes, numNodes])
        self.nodes = nodes
        self.edges = edges
        self.colors = None
        self.clusterCenters = np.array([])

    def init_adjacency_list(self, edges):
        for edge1, edge2 in edges:
            self.adjacency_list[edge1, edge2] = self.adjacency_list[edge2, edge1] = 1

    def fill_adjacency_list(self, adj_list, nodeList):
        for x in range(len(nodeList)):
            for y in range(len(nodeList)):
                adj_list[x,y] = self.adjacency_list[nodeList[x],nodeList[y]]

    def page_rank(self, adjacency_list, numNodes):
        damping_factor = 0.85
        A = adjacency_list
        d = np.dot(A,np.ones([numNodes,1]))
        D = np.identity(numNodes) * d
        P = np.dot(A,np.linalg.pinv(D))
        colors = np.dot((1 - damping_factor)*np.linalg.inv(np.identity(numNodes)-damping_factor*P),np.ones([numNodes,1])/numNodes)
        return colors