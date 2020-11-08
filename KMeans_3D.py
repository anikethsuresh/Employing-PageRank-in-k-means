import numpy as np
import matplotlib.pyplot as plt
import trimesh
import networkx as nx
from KMeansWithPageRank import KMeansWithPageRank
from Graphing import *

mesh = trimesh.load("Labrador retriever.glb", force="mesh")
mesh.visual = trimesh.visual.color.ColorVisuals()

# testMesh = trimesh.Trimesh(vertices=[[0,0,0],[0,0,2],[0,0,1],[0,1,0],[0,1,2],[0,1,1],[1,0,0],[1,0,2],[1,0,1],[1,1,0],[1,1,2],[1,1,1]],
#                         faces=[[9,6,0],[9,0,3],[11,8,2],[11,2,5],[10,7,1],[10,1,4],[5,11,9],[5,9,3],[4,10,11],[4,11,5],[9,11,8],[9,8,6],[11,10,7],[11,7,8],
#                         [4,5,2],[4,2,1],[5,3,0],[5,0,2],[0,6,8],[0,8,2],[2,8,7],[2,7,1]])

# No idea why but this is how you should set the colors for the faces 
# indices = list(range(testMesh.visual.vertex_colors.shape[0]))
# testMesh.visual.vertex_colors[indices] = trimesh.visual.random_color()
mesh.show()
G = mesh.vertex_adjacency_graph
nodes = {}
for i in range(mesh.vertices.shape[0]):
        nodes[i] = tuple(mesh.vertices[i])
edges = list(G.edges)
myGraph = My3DGraph(G, mesh, len(nodes), nodes, edges)
myGraph.init_adjacency_list(edges)
# myGraph.show("PageRank over the entire graph", colors)

KMeansWithPageRank(myGraph, 2, mesh3D=True)
