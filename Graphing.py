"""
Graphing library to work with 2D points, 2D Graphs and 3D Meshes
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import trimesh

class MyPoints():
    """
    Class used to represent points in euclidean space
    """
    def __init__(self):
        """
        Initialize the graph, colors of each node and the center of the clusters
        """
        self.network = None
        self.color = None
        self.clusterCenters = None
    
    def setNetwork(self, network, color):
        """
        Setter

        Arguments:
        network - graph (networkx) 
        color - colors of the nodes to represent the cluster to which it belongs to
        """
        self.network = network
        self.color = color
    
    def show(self, title, withCenters = False, final=False):
        """
        Visualizes the graph with the colors representing the cluster to which it belongs

        Arguments:
        title - title of the matplotlib plot
        withCenters - bool, whether or not to show the centers in the plot
        final - bool, whether this is the final plot. Clustering complete
        """
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
    """
    My own graph to hold objects for graph manipulation in 2D
    """
    def __init__(self, networkxGraph, numNodes, nodes, edges):
        """
        Initialize the graph, colors of each node and the center of the clusters

        Arguments:
        networkxGraph - graph from the networkx library
        numNodes - int, number of nodes/vertices in the graph
        nodes - dict, keys are the nodes and the values are its coordinates in 2D space
        edges - list, list of edges between nodes
        """
        self.graph = networkxGraph
        self.numNodes = numNodes
        self.adjacency_list = np.zeros([numNodes, numNodes])
        self.nodes = nodes
        self.edges = edges
        self.colors = None
        # Initialize the adjacency list
        self.init_adjacency_list(self.edges)
        self.clusterCenters = np.array([])


    def init_adjacency_list(self, edges):
        """
        Initialize the adjacency with the edges of the graph
        
        Arguments:
        edges - list, list of edges between nodes
        """
        for edge1, edge2 in edges:
            self.adjacency_list[edge1, edge2] = self.adjacency_list[edge2, edge1] = 1


    def fill_adjacency_list(self, adj_list, nodeList):
        """
        Fill adjacency list with the nodes in nodeList

        Arguments:
        nodeList - list, list of nodes
        """
        # adj_list = self.adjacency_list[np.ix_(nodeList, nodeList)]
        for x in range(len(nodeList)):
            for y in range(len(nodeList)):
                adj_list[x,y] = self.adjacency_list[x,y]
                

    def show(self, title, colors, withCenters=False, final=False, showEdges=True):
        """
        Visualizes the graph with the colors representing the cluster to which it belongs

        Arguments:
        title - title of the matplotlib plot
        colors - colors of the nodes to represent the cluster to which it belongs to
        withCenters - bool, whether or not to show the centers in the plot
        final - bool, whether this is the final plot. Clustering complete
        showEdges - bool, whether the edges should be shown 
        
        """

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
        """
        Calculate the PageRank vector

        Arguments:
        adjacency_list - adjacency list of the graph
        numNodes - int, number of nodes in the graph

        Returns the colors of the nodes to represent the cluster to which it belongs to
        """
        damping_factor = 0.85
        A = adjacency_list
        d = np.dot(A,np.ones([numNodes,1]))
        D = np.identity(numNodes) * d
        # Since we are dealing with square matrices, most of them do not possess an inverse (singular matrix),
        # Therefore, to account for this, we take their pseudo inverse instead
        P = np.dot(A,np.linalg.pinv(D))
        colors = np.dot((1 - damping_factor)*np.linalg.inv(np.identity(numNodes)-damping_factor*P),np.ones([numNodes,1])/numNodes)
        return colors

class My3DGraph(MyGraph):
    """
    My own graph to hold objects for graph manipulation in 3D. Inherits from MyGraph
    """

    def __init__(self, networkxGraph, trimeshMesh, numNodes, nodes, edges, actualVertices):
        """
        Arguments:
        networkxGraph - graph from the networkx library
        trimeshMesh - mesh from the trimesh library
        numNodes - int, number of nodes/vertices in the graph
        nodes - dict, keys are the nodes and the values are its coordinates in 2D space
        edges - list, list of edges between nodes
        actualVertices - dict, keys are the vertex as represented in trimesh, and the values are their actual vertex id
                        to ensure that no vertex is repeated
        """
        super().__init__(networkxGraph, numNodes, nodes, edges)
        self.mesh = trimeshMesh
        self.actualVertices = actualVertices
        # Color palatte to color the mesh. Currently supports up to 6 clusters
        self.color_palatte = [[249, 65, 68, 255],[243, 114, 44, 255],[248, 150, 30, 255],[249, 199, 79, 255],[144, 190, 109, 255],[67, 170, 139, 255],[87, 117, 144, 255]]
    
    def show(self, colors):
        """
        Visualizes the graph with the colors representing the cluster to which it belongs

        Arguments:
        colors - colors of the nodes to represent the cluster to which it belongs to
        """
        # Due to some issue with trimesh, the vertex_colors do not update when called in quick succession
        # As a result, I set the color of the vertices twice. This works.
        for idkwhy in range(2):
            for i in range(self.clusterCenters.shape[0]):
                indices = np.where(colors == i)[0]
                self.mesh.visual.vertex_colors[indices] = self.color_palatte[i]
        self.mesh.show()
        