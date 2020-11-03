"""
Main program to implement k-means
"""
from Graphing import *
from Kmeans import *
from sklearn import datasets

if __name__ == "__main__":
    # circles, colors = datasets.make_circles(n_samples=1500, noise=0.05)
    circles, colors = datasets.make_blobs(n_samples=1500, random_state=8)
    graph = Graph()
    graph.setNetwork(circles, colors)
    # graph.show()
    kmeans = KMeans(graph, 3)