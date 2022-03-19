# K-Nearest Neighbors (K-NN)
from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

for X_set, y_set in zip(X_train,y):
    if y_set == 1:
        plt.scatter(X_set[0],X_set[1],c ='blue',label='1')
    else:
        plt.scatter(X_set[0],X_set[1],c ='red',label='2')
if y_pred[0] == 0:
    plt.plot(X_test[0,0],X_test[0,1],'b^', color='red', markersize = 8)
else :
    plt.plot(X_test[0,0],X_test[0,1],'b^', color='blue', markersize = 8)

plt.show()