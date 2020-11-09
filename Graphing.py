"""
Graphing library
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import trimesh

class MyPoints():
    def __init__(self):
        self.network = None
        self.color = None
        self.clusterCenters = None
    
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
        self.init_adjacency_list(self.edges)
        self.clusterCenters = np.array([])

    def init_adjacency_list(self, edges):
        for edge1, edge2 in edges:
            self.adjacency_list[edge1, edge2] = self.adjacency_list[edge2, edge1] = 1

    def fill_adjacency_list(self, adj_list, nodeList):
        adj_list = self.adjacency_list[np.ix_(nodeList, nodeList)]
                
    def show(self, title, colors, withCenters=False, final=False, showEdges=True):
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        if showEdges:
            nx.draw_networkx_edges(self.graph,self.nodes,alpha=0.4, edge_color="#424242")
        nx.draw_networkx_nodes(self.graph,self.nodes, node_color=list(colors),cmap=plt.cm.jet,
                       node_size=20)
        # Debugging: colors the first node    
        # nx.draw_networkx_nodes(self.graph,self.nodes, nodelist = [0],cmap=plt.cm.jet,node_color="#ff1cf3",
        #                node_size=20)
        # Debugging: prints the labels with the nodes
        # nx.draw_networkx_labels(self.graph,self.nodes, {x:str(x) for x in self.nodes.keys()}, font_size=4)

        if withCenters:
            nx.draw_networkx_nodes(self.graph,self.nodes,nodelist =list(self.clusterCenters),node_color="#ff1c03",node_shape="X",
                        node_size=80)
        plt.pause(1)
        if final:
            plt.show()
        else:
            plt.clf()

    def page_rank(self, adjacency_list, numNodes):
        damping_factor = 0.85
        A = adjacency_list
        d = np.dot(A,np.ones([numNodes,1]))
        D = np.identity(numNodes) * d
        P = np.dot(A,np.linalg.pinv(D))
        colors = np.dot((1 - damping_factor)*np.linalg.inv(np.identity(numNodes)-damping_factor*P),np.ones([numNodes,1])/numNodes)
        return colors

class My3DGraph(MyGraph):
    def __init__(self, networkxGraph, trimeshMesh, numNodes, nodes, edges, actualVertices):
        super().__init__(networkxGraph, numNodes, nodes, edges)
        self.mesh = trimeshMesh
        self.actualVertices = actualVertices
        self.color_palatte = [[249, 65, 68, 255],[243, 114, 44, 255],[248, 150, 30, 255],[249, 199, 79, 255],[144, 190, 109, 255],[67, 170, 139, 255],[87, 117, 144, 255]]
    
    def show(self, title, colors):
        # Due to some issue with trimesh, the vertex_colors do not update when called in quixck succession
        # As a result, I set the color of the vertices twice. This works.
        for idkwhy in range(2):
            for i in range(self.clusterCenters.shape[0]):
                indices = np.where(colors == i)[0]
                self.mesh.visual.vertex_colors[indices] = self.color_palatte[i]
        self.mesh.show()
        