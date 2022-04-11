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

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',random_state=0)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
#visual
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

xy = np.array([X1.ravel(), X2.ravel()]).T
P = model.predict(xy).reshape(X1.shape)

plt.contourf(X1,X2, P, alpha=0.75, cmap = plt.cm.Set1)

plt.scatter(X_train[:,0],X_train[:,1],c=y_train, cmap=plt.cm.Set1)

plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()
