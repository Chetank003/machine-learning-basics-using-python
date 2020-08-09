# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:35:22 2019

@author: chethan.k
"""
#importing libbraries
#import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#initialising the data and model
iris = load_iris()
logistic_regressor = LogisticRegression()

#splitting the data and training the model
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2)
logistic_regressor.fit(X_train, y_train)

#predicting the values using the trained model
y_pred = logistic_regressor.predict(X_test)
logistic_regressor_score = logistic_regressor.score(X_test, y_test)

#plotting actual vs predicted values
plot_data_iris = pd.DataFrame({'Actual': y_test,'Predicted': y_pred})
plot_data_iris.plot(kind = 'bar', figsize = (10, 10))
plt.title("Iris data Actual vs Prediction(plot - 1)")
plt.show()