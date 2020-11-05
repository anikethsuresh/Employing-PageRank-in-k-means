"""
Graphing library
"""
import numpy as np
import matplotlib.pyplot as plt

class Graph():
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

class Point():
    def __init__(self,x,y,label):
        self.x = x
        self.y = y
        self.label = label