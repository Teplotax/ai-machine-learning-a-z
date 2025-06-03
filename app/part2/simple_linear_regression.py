##############################
# Importing the libraries
##############################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

##############################
# Importing the dataset
##############################
dataset = pd.read_csv('dataset/salary_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

############################################################
# Splitting the dataset into the Training set and Test set
############################################################
# from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print('====================')
print(X_train)
print('====================')
print(X_test)
print('====================')
print(y_train)
print('====================')
print(y_test)

