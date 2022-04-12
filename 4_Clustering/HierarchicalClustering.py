import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# use dendrogram to find the optimal number of clusters
""" 
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

plt.title('Dendrogram')
plt.xlabel('Customer')
plt.ylabel('Eclidean distances')
plt.show()

tuong tu ebow trong kmeans
"""

from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = model.fit_predict(X)


# visual
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1],s=100, c='red', label='1')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1],s=100, c='blue', label='2')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1],s=100, c='green', label='3')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1],s=100, c='black', label='4')
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1],s=100, c='yellow', label='5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

