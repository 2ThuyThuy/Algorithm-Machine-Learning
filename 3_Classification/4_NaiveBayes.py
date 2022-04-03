import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets = pd.read_csv('Social_Network_Ads.csv')
X_data = datasets.iloc[:,[2,3]].values
y_data = datasets.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data,y_data, test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.naive_bayes import GaussianNB #naive_bayes
model = GaussianNB()
model.fit(X_train, y_train)


# visual
plt.scatter(X_train[:,0],X_train[:,1],c=y_train, cmap=plt.cm.Set1)


X1, Y1 = np.meshgrid(np.arange(start = X_train[:, 0].min() - 1, stop = X_train[:, 0].max() + 1, step = 0.01),
                   np.arange(start = X_train[:, 1].min() - 1, stop = X_train[:, 1].max() + 1, step = 0.01))

xy = np.vstack([X1.ravel(),Y1.ravel()]).T
P = model.predict(xy).reshape(X1.shape)

plt.contour(X1,Y1, P, alpha=0.75, cmap=plt.cm.Set1)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()
