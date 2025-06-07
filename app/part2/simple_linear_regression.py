##############################
# Importing the libraries
##############################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

# print('====================')
# print(X_train)
# print('====================')
# print(X_test)
# print('====================')
# print(y_train)
# print('====================')
# print(y_test)

############################################################
# Training the Simple Linear Regression model on the Training set
############################################################
# from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


############################################################
# Predicting the Test set results
############################################################
y_pred = regressor.predict(X_test)

# print(y_test)
# print('====================')
# print(y_pred)

############################################################
# Visualising the Training set results
############################################################
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# R² for Linear Regression
r2 = regressor.score(X, y)
print(f"R² for Linear Regression: {r2:.4f}")