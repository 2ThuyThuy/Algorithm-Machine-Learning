import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1],s=100, c='red', label='1')
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1],s=100, c='blue', label='2')
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1],s=100, c='green', label='3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.show()