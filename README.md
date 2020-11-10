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
    - [trimesh](https://github.com/mikedh/trimesh): Trimesh is a pure Python (2.7-3.4+) library for loading and using triangular meshes.
    
    
### Running the program
Download all the libraries in the requirements file using 'pip install -r requirements.txt'
Running the files in this order, individually show the entire picture (that I'm trying to show)
1. KMeans_Naive.py: In cases where this cannot be seperated using naive k-means it fails as below:
<img src="Output images/moons_kmeans_naive.png">
2. KMeans.py: We can see that it is not able to seperate the two clusters using PageRank as the update mechanism
<img src="Output images/moons_kmeans_pagerank.png">
3. KMeans_3D.py: Applying the same PageRank method to meshes, we can see that it is possible to achieve considerable clustering. While the termination criteria is not clear, the different parts of the body are segregated.
<img src="Output images/bear1.png" width="300" height="600" display:"inline-block" >
<img src="Output images/dog.png"  display:"inline-block">
<img src="Output images/dolphin3.png" width="600" height="600" display:"inline-block">
