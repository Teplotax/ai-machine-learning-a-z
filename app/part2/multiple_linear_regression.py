import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Importing the dataset
# import pandas as pd
dataset = pd.read_csv('dataset/50_startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print(X)

# Encoding categorical data
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# import numpy as np
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# print(X)

# Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
# from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)



# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)

# How do I use my multiple linear regression model to make a single prediction
# X_single = [[66051.52, 182645.56, 118148.2, 'Florida']]
# X_single = ct.transform(X_single)
# y_single_pred = regressor.predict(X_single)
# print(y_single_pred)







