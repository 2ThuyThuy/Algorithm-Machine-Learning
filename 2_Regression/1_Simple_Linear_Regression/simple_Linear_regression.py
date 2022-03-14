import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('D:\\Study\\ML\\Algorithm-Machine-Learning\\2_Regression\\data\\Salary_Data.csv')
#print(dataset.head())
#print(dataset.info())
X = dataset.iloc[:,:-1].values # return array 2D
y = dataset.iloc[:,1].values
#print(X,y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
#print("Coefficients",model.coef_)

y_predict = model.predict(X_test)

# Visualising result
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, model.predict(X_train))
plt.title("Salary vs Exp")
plt.xlabel("Years of exp")
plt.ylabel("Salary")
plt.show()
