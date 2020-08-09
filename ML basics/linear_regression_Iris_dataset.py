# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:18:36 2019

@author: chethan.k
"""
#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#data initialization and model creation
iris = load_iris()
linear_regressor = LinearRegression()
kfold = KFold(n_splits = 5, shuffle = True, random_state = 0)
iris_regressor_score = []

#traing linear_regressor model with KFold
for train, test in kfold.split(iris.data):
    X_train, X_test, y_train, y_test = iris.data[train], iris.data[test], iris.target[train], iris.target[test]
    linear_regressor.fit(X_train, y_train)
    print("Training Score: ",linear_regressor.score(X_test, y_test),"\n")
    iris_regressor_score.append(linear_regressor.score(X_test, y_test))
    #pickle.dump(linear_regressor,"linear_regressor.h5")

#plotting the trained model
plt.plot(iris.target_names[iris.target], iris.target, color = 'red', label = "Target")
plt.scatter(iris.target_names[iris.target], linear_regressor.predict(iris.data), label = "Predicted")
plt.title("Iris_prediction(Scattered) vs Iris_target(Plot_line) on total data")
plt.xlabel("Iris Types")
plt.ylabel("Iris target values")
plt.legend()
plt.show()

#predicting using test data
y_pred = linear_regressor.predict(X_test)

#plotting the trained model on test data
plt.plot(iris.target_names[y_test], y_test, color = 'red', label = "y_test")
plt.scatter(iris.target_names[y_test], linear_regressor.predict(X_test), label = "y_pred")
plt.title("Iris_prediction(Scattered) vs Iris_target(Plot_line) on test data")
plt.xlabel("Iris Types")
plt.ylabel("Iris target values")
plt.legend()
plt.show()

#plotting actual vs predicted values
plot_data_iris = pd.DataFrame({'Actual': y_test,'Predicted': y_pred})
plot_data_iris.plot(kind = 'bar', figsize = (10, 10))
plt.title("Iris data Actual vs Prediction(plot - 1)")
plt.show()

x_axis = np.array(range(0,y_test.shape[0]))
plt.scatter(x_axis, y_test, label = "y_test", color = "r")
plt.plot(x_axis, y_pred, label = "y_pred", color = "b")
plt.title("Iris data Actual vs Prediction(plot - 2)")
plt.legend()
plt.show()

#Evaluating model parameters
print("Model Evaluations: \n")
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("Score: ", linear_regressor.score(X_test, y_test))