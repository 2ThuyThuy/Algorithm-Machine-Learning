import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('D:\\Study\\ML\\Algorithm-Machine-Learning\\2_Regression\\data\\Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

# training
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=10, random_state=0)
model.fit(X,y)



plt.scatter(6.5, model.predict([[6.5]]), c='b')
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.plot(X_grid,model.predict(X_grid), c='b')
plt.scatter(X, y, c='r')
plt.title('Truth or Bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()