import numpy as np
import matplotlib.pyplot as plt
import trimesh
import networkx as nx
from KMeansWithPageRank import KMeansWithPageRank
from Graphing import *


def getEdgesInMyFormat(mesh, nodes):
    actual = {}
    check = [True] * len(nodes)
    verts = mesh.vertices
    for i in range(len(nodes)):
        if check[i]:
            add_to_dict = np.where(verts[:, 0] == verts[i, 0])[0]
            check[i] = False
            current = add_to_dict[0]
            for j in add_to_dict:
                check[j] = False
                actual[j] = current
    return actual


if __name__ == "__main__":
    # Load the mesh
    mesh = trimesh.load("Bear - low poly.glb", force="mesh")
    # Loaded mesh automatically has a texture for the faces. Set this to a ColorVisual. Allows you to play with the
    # colors of the mesh
    mesh.visual = trimesh.visual.color.ColorVisuals()
    mesh.show()
    # Get the networkx graph
    G = mesh.vertex_adjacency_graph
    nodes = {}
    # Get the nodes in tuple form
    for i in range(mesh.vertices.shape[0]):
        nodes[i] = tuple(mesh.vertices[i])
    edges = np.array(G.edges)
    actualEdges = getEdgesInMyFormat(mesh, nodes)

    newEdges = edges.copy()
    # Get the keys in the format I want it in, where all the edges are connected in the graph. Since the internal represenatation
    # does not repeat vertices, the entire graph is not connected. The code below fixes this
    for key in actualEdges.keys():
        newEdges = np.where(newEdges == key, actualEdges[key], newEdges)
        # Create a new graph to house the new edges (as this is used in the distance calculation)
    newG = nx.Graph()
    newG.add_edges_from(list(newEdges))
    myGraph = My3DGraph(newG, mesh, len(nodes), nodes, newEdges, actualEdges)
    myGraph.init_adjacency_list(newEdges)

    KMeansWithPageRank(myGraph, 6)
