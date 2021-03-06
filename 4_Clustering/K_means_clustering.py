import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# use Elbow to slove K 
""" 
from yellowbrick.cluster.elbow import KElbowVisualizer
model = KElbowVisualizer(KMeans(),k=10)
model.fit(X)
model.show()
"""


plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1],s=100, c='red', label='1')
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1],s=100, c='blue', label='2')
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1],s=100, c='green', label='3')
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1],s=100, c='black', label='4')
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1],s=100, c='yellow', label='5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.show()
