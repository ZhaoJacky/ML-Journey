import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Turns the csv file into a pandas Dataframe.
dataset = pd.read_csv('simplelineardata.csv')

# Organizes the data into features (X) and targets (y)
X = dataset.iloc[:, [0]].values # The brackets around 0 keep it as a 2D dataframe.
y = dataset.iloc[:, -1].values # I believe targets should be 1D arrays.

# Split the data into a training set (80%) and a test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create an instance of the Linear Regression class and trains the model using .fit(...).
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Use the trained model to predict the targets using X_test
y_pred = regressor.predict(X_test)

# Reshapes the 1D arrays from horizontal to vertical.
# In more technical terms, reshapes from 1D to 2D Column Vectors
y_pred = y_pred.reshape(len(y_pred), 1)
y_test = y_test.reshape(len(y_test), 1)

# Concatenate 2D arrays horizontally column-wise (indicated by the parameter 1).
compare = np.concatenate((y_pred, y_test), 1)

# Print the concatenated arrays to compare the actual targets and predicted targets.
print(compare)




