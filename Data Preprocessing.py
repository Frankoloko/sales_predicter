# Data Preprocessing

# Importing the library

import numpy as sNumpy
import matplotlib.pyplot as sPyplot
import pandas as sPandas

# Importing the dataset
dataset = sPandas.read_csv('Data Preprocessing.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Fix missing data by using mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Transform categorical data into numbers
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Transform non-progressive categorical numbers into multiple columns
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Slitting the dataset into training and testing datasets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Scale all number values to the same range
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # Impportant here you dont add fit_ 7:00 => https://www.udemy.com/course/machinelearning/learn/lecture/5683432#questions


X = sPandas.DataFrame(X)
Y = sPandas.DataFrame(Y)