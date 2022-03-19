from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load data
path = "D:\\Study\\ML\\Algorithm-Machine-Learning\\3_Classification\\data\\Social_Network_Ads.csv"
dataset = pd.read_csv(path)
X = dataset.iloc[:,[2,3]].values # get age
y = dataset.iloc[:,4].values  # get boolean

#print(dataset.info())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
#visual 
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01),np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('tomato', 'lightgreen')))
plt.title('predict result')
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c = ListedColormap(('red', 'green'))(i), label = j)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()