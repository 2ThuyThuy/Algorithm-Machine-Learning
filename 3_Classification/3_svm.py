import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


dataset = datasets.load_iris()
X = dataset.data[:, :2]
y = dataset.target
y =[ 1 if i > 0 else 0 for i in y]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

from sklearn.svm import SVC
model = SVC(kernel='linear', random_state=0)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(y_test, y_pred))


plt.figure(2, figsize=(8, 6))
plt.clf()

X1, X2 = np.meshgrid(np.arange(start = X_train[:, 0].min() - 1, stop = X_train[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_train[:, 1].min() - 1, stop = X_train[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75,  cmap=plt.cm.Set1 )


plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Set1, edgecolor="k")
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, cmap=plt.cm.Set3, edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.xticks(())
plt.yticks(())
plt.show()
