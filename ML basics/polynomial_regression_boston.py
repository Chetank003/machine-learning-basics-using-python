# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:11:34 2019

@author: chethan.k
"""
#TODO: Applying linear regression on boston dataset yeilded 
#Mean Squared Error:  16.573690392314706 and Score:  0.8395331660532515
#so in this we try to improve the score by transforming original features to polynomial features

#importing datasets
import numpy as np
#import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

#data and model initialization
boston_poly = load_boston()
polynomial_features = PolynomialFeatures(degree = 2)
kfold_poly = KFold(n_splits = 5, shuffle = True, random_state = 0)
linear_regressor_poly = LinearRegression()
boston_poly_regressor_score = []

#training the model with KFold and polynomial_features
for train, test in kfold_poly.split(boston_poly.data):
    X_train, X_test, y_train, y_test = boston_poly.data[train], boston_poly.data[test], boston_poly.target[train], boston_poly.target[test]
    linear_regressor_poly.fit(X_train, y_train)
    y_pred = linear_regressor_poly.predict(X_test)
    boston_poly_regressor_score.append(linear_regressor_poly.score(X_test, y_test))

#final training score of the model
print("Training r2 Score: ",r2_score(y_test, y_pred)) 

#plotting trained model on test data(without transformed data)
x_axis = np.array(range(0,y_test.shape[0]))
plt.plot(x_axis, y_test, color = "r", label = 'y_test')
plt.plot(x_axis, y_pred, color = "b", label = 'y_pred')
plt.title("plotting trained model on test data(without transformed data)")
plt.legend()
plt.show()   

#transforming the existing features to higher(polynomial) features
X_train_poly = polynomial_features.fit_transform(X_train)
X_test_poly = polynomial_features.fit_transform(X_test)

#fitting the linear regressor with transformed data
linear_regressor_poly.fit(X_train_poly, y_train)

#determining the r2_score of the model on the training data
y_pred_train = linear_regressor_poly.predict(X_train_poly)
print("\n\nTraining r2 Score with polynomial features: ",r2_score(y_train, y_pred_train))

#plotting trained model on training data(transformed data)
x_axis = np.array(range(0,y_train.shape[0]))
plt.plot(x_axis, y_train, color = "r", label = 'y_train')
plt.plot(x_axis, y_pred_train, color = "b", label = 'y_pred_train')
plt.title("plotting trained model on training data(transformed data)")
plt.legend()
plt.show()

#determining the r2_score of the model on the test data
y_pred_test = linear_regressor_poly.predict(X_test_poly)
print("\n\nTesting r2 Score with polynomial features: ",r2_score(y_test, y_pred_test))

#plotting trained model on test data(transformed data)
x_axis = np.array(range(0,y_test.shape[0]))
plt.plot(x_axis, y_test, color = "r", label = 'y_test')
plt.plot(x_axis, y_pred_test, color = "b", label = 'y_pred_test')
plt.title("plotting trained model on test data(transformed data)")
plt.legend()
plt.show()