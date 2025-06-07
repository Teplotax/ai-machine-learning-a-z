##############################
# Importing the libraries
##############################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

##############################
# Importing the dataset
##############################
dataset = pd.read_csv('dataset/data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print(X)
# print(y)

##############################
# Taking care of missing data
##############################
# from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # replaces nan values for the column mean
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# print(X)

#########################################
# Encoding the Independent Variable
#########################################
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# print(X)

#################################
# Encoding the Dependent Variable
#################################
# from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

# print(y)

############################################################
# Splitting the dataset into the Training set and Test set
############################################################
# from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# print('====================')
# print(X_train)
# print('====================')
# print(X_test)
# print('====================')
# print(y_train)
# print('====================')
# print(y_test)

############################################################
# Feature Scaling
############################################################
# from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:]) #The [:, 3:] removes the one-hot encoded columns
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print('====================')
print(X_train)
print('====================')
print(X_test)