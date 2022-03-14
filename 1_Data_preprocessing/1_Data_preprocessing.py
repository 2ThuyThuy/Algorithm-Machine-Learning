import numpy as np
import pandas as pd

# import data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values # return array 2D
y = dataset.iloc[:,3].values

# missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:,1:3]) # fit the imputer on X[:1,3]
X[:,1:3] = imputer.transform(X[:,1:3]) # missint data column 1 -> 3
#print(X)

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
LabelEncoder_X = LabelEncoder()
X[:,0] = LabelEncoder_X.fit_transform(X[:,0]) # convert Country to  number 
ct = ColumnTransformer([('Country', OneHotEncoder(),[0])],remainder="passthrough") # split 
#passthrough save different columns when don't use transformer
X = ct.fit_transform(X)
#print(X)
# same with y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) # convert "yes", "no" -> 1,0

#splitting the dataset into the training set and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler()
X_train[:,3:] = sc_x.fit_transform(X_train[:,3:]) 
X_test[:,3:] = sc_x.transform(X_test[:,3:])

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
print(X_train)
print(X_test)