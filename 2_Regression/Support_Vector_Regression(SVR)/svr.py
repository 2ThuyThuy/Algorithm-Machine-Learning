import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1:].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
y = sc.fit_transform(y)

from sklearn.svm import SVR
model = SVR(kernel='rbf')
model.fit(X,y)

print(model.predict(X))

plt.plot(X,model.predict(X),c='blue')
plt.scatter(X,y,c='green')
plt.scatter(X,y, c='r')
plt.plot(X,y, c='red')
plt.show()