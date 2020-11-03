"""
Main program to implement k-means
"""
from Graphing import *
from Kmeans import *
from sklearn import datasets

if __name__ == "__main__":
    circles, colors = datasets.make_circles()
    graph = Graph()
    graph.setNetwork(circles, colors)
    # graph.show()
    kmeans = KMeans(graph, 3)