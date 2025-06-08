import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
# import pandas as pd
dataset = pd.read_csv('dataset/position_salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset
# from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)



# Training the Polynomial Regression model on the whole dataset
# from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)



# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Finer polynomial plot
# Generate a finer grid of X values (0.1 steps)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# How do I use my polynomial regression model to make a single prediction
X_single = [[6]]
X_single_poly = poly_reg.transform(X_single)  # Transform to polynomial features
prediction = lin_reg_2.predict(X_single_poly)
print(f"Predicted salary for level 6: {prediction[0]}")


# R² for Linear Regression
r2_linear = lin_reg.score(X, y)
print(f"R² for Linear Regression: {r2_linear:.4f}")

# R² for Polynomial Regression
r2_poly = lin_reg_2.score(X_poly, y)
print(f"R² for Polynomial Regression: {r2_poly:.4f}")

# R² for Polynomial Regression with sklearn
from sklearn.metrics import r2_score
y_pred = lin_reg_2.predict(X_poly)
print(r2_score(y, y_pred))