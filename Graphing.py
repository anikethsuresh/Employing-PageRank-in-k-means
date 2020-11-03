"""
Graphing library
"""
import numpy as np
import matplotlib.pyplot as plt

class Graph():
    def __init__(self):
        print("Graph created")
        self.network = np.array([])
        self.color = np.array([])

    def addPoint(self, point, color):
        self.network = np.vstack((self.network,point))
        self.color = np.vstack((self.color,color))
    
    def setNetwork(self, network, color):
        self.network = network
        self.color = color
    
    def show(self):
        plt.scatter(self.network[:,0],self.network[:,1], c=self.color)
        plt.show()

class Point():
    def __init__(self,x,y,label):
        self.x = x
        self.y = y
        self.label = label