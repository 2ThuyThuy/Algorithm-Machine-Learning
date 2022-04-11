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

from sklearn.svm import SVC 
model = SVC(kernel='rbf', random_state=0)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

#draw 

def plot_svm(model,ax=None, plot_support=True, X_train=[]):
    if ax == None:
        ax = plt.gca()

    X, Y = np.meshgrid(np.arange(start = X_train[:, 0].min() - 1, stop = X_train[:, 0].max() + 1, step = 0.01),
                       np.arange(start = X_train[:, 1].min() - 1, stop = X_train[:, 1].max() + 1, step = 0.01))


    xy = np.vstack([X.ravel(),Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(X,Y,P, colors='k', levels=[-1, 0, 1], alpha = 0.5, linestyles=['--','-','--'])

    if plot_support:
        ax.scatter(model.support_vectors_[:,0],
                   model.support_vectors_[:,1],
                   c='none',
                   s=300,linewidths=1, edgecolors='k')

plt.scatter(X_train[:,0],X_train[:,1],c=y_train, cmap=plt.cm.Set1)
#plt.scatter(X_test[:,0],X_test[:,1],c=y_test, cmap=plt.cm.Set2)
plot_svm(model,plot_support=False,X_train=X_train)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')

plt.show()