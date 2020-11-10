# Employing-PageRank-in-k-means

The following project uses kmeans to perform clustering by using PageRank as the update mechanism.

### Directory Structure
1. **Meshes**: 3D obj meshes that can be used to run K_Means_3D.py
2. **Output images**: set of images that show the output window when the programs are run.
3. **Related papers**: a few papers I found with suitable material to calculate the PageRank vector.
4. *DistanceMetric.py*: Class to perform distance calculation between nodes, points and vertices
5. *Graphing.py*: Graphing library to work with 2D points, 2D Graphs and 3D Meshes.
6. *KMeans_3D.py*: Implements the k-means algorithm with PageRank in a 3D Mesh.
7. *KMeans_Naive.py*: Implements the naive k-means algorithm (used the **mean** to update clusters).
8. *KMeans.py*: Implements the k-means algorithm with PageRank in a 2D Mesh.
9. *KMeansWithPageRank.py*: main class which provides the skeleton to perform k-means using PageRank.
10. **requirements.txt**: requirements text file with the different libraries used in the project. In reality, the main libraries would be:
    - [numpy](https://numpy.org/): Numerical computing tools
    - [networkx](https://networkx.org/): Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
    - [matplotlib](https://matplotlib.org/): Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. 
    
    
### Running the program
Download all the libraries in the requirements file using 'pip install -r requirements.txt'
Running the files in this order, individually show the entire picture (that I'm trying to show)
1. KMeans_Naive.py
![[Output images/moons_kmeans_naive.png]]
