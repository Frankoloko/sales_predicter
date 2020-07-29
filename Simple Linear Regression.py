# Data Preprocessing

# Importing the library

import numpy as sNumpy
import matplotlib.pyplot as sPyplot
import pandas as sPandas

# Importing the dataset
dataset = sPandas.read_csv('Simple Linear Regression.csv')
X = dataset.iloc[:, 2].values.reshape(-1, 1) # This reshape fixes the error "Expected 2D array, got 1D array instead:"
Y = dataset.iloc[:, 3].values.reshape(-1, 1) # This reshape fixes the error "Expected 2D array, got 1D array instead:"

# Slitting the dataset into training and testing datasets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set results
y_predicted = regressor.predict(X_test)

# Visualizing the training set results
sPyplot.scatter(X_train, Y_train, color = 'red')
sPyplot.plot(X_train, regressor.predict(X_train), color = 'blue')
sPyplot.title('Salary vs Age (training set)')
sPyplot.xlabel('Age')
sPyplot.ylabel('Salary')
sPyplot.show()

# Visualizing the test set results
sPyplot.scatter(X_test, Y_test, color = 'red')
sPyplot.plot(X_train, regressor.predict(X_train), color = 'blue') # IMPORTANT: you dont replace the train with test in this line
sPyplot.title('Salary vs Age (test set)')
sPyplot.xlabel('Age')
sPyplot.ylabel('Salary')
sPyplot.show()

X = sPandas.DataFrame(X)
Y = sPandas.DataFrame(Y)