# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 08:44:53 2019

@author: chethan.k
"""
#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#data initialization and model creation
boston = load_boston()
linear_regressor = LinearRegression()
kfold = KFold(n_splits = 5, shuffle = True, random_state = 0)
boston_regressor_score = []

#traing linear_regressor model with KFold
for train, test in kfold.split(boston.data):
    X_train, X_test, y_train, y_test = boston.data[train], boston.data[test], boston.target[train], boston.target[test]
    linear_regressor.fit(X_train, y_train)
    print("Training Score: ",linear_regressor.score(X_test, y_test),"\n")
    boston_regressor_score.append(linear_regressor.score(X_test, y_test))
    #pickle.dump(linear_regressor,"linear_regressor.h5")

#predicting using test data
y_pred = linear_regressor.predict(X_test)

#plotting the trained model
plot_data_boston = pd.DataFrame({'Actual': y_test[:20],'Predicted': y_pred[:20]})
plot_data_boston.plot(kind = 'bar', figsize = (10, 10))
plt.title("Boston housing data Actual vs Prediction (only 20 values)")
plt.show()

x_axis = np.array(range(0,y_test.shape[0]))
plt.plot(x_axis, y_test, label = "y_test", color = "r")
plt.plot(x_axis, y_pred, label = "y_pred", color = "b")
plt.legend()
plt.show()

'''
#plotting the trained model on test data
plt.plot(iris.target_names[y_test], y_test, color = 'red')
plt.scatter(iris.target_names[y_test], linear_regressor.predict(X_test))
plt.title("Iris_prediction(Scattered) vs Iris_target(Plot_line) on test data")
plt.xlabel("Iris Types")
plt.ylabel("Iris target values")
plt.show()
'''

#Evaluating model parameters
print("Model Evaluations: \n")
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("Score: ", linear_regressor.score(X_test, y_test))