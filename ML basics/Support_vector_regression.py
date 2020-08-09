# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:00:37 2019

@author: chethan.k
"""
#importing libraries
#import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#importing data and initialising features and targets
data = pd.read_csv("Salary_Data.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,1].values

#splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#initialising and training SVR
sv_regressor = SVR()
sv_regressor.fit(X_train, y_train)

#predicting the output
y_pred = sv_regressor.predict(X_test)

#visualising the output
plt.scatter(X, y, label = "Actual data")
plt.plot(X, sv_regressor.predict(X), label = "Regressor line")
plt.legend()
plt.show()

'''
Since SVR does not have any internal feature scaling the model is inaccurate
here we normalise both data and target so that they are scaled on same level
we can also use StandardScalar from sklearn.preprocessing for this
'''
#X_scaled = (X - X.mean()) / (X.max() - X.min())
#y_scaled = (y - y.mean()) / (y.max() - y.min())
standardscaler_x = StandardScaler()
standardscaler_y = StandardScaler()
X_scaled = standardscaler_x.fit_transform(X)
y_scaled = standardscaler_y.fit_transform(y.reshape(-1,1))

#splitting the scaled data into training and test sets
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X, y, test_size = 0.2, random_state = 0)

#initialising and training SVR for scaled data
sv_regressor_scaled = SVR()
sv_regressor_scaled.fit(X_train_scaled, y_train_scaled)

#predicting the output
y_pred_scaled = sv_regressor_scaled.predict(X_test_scaled)

#visualising the output
plt.scatter(X_scaled, y_scaled, label = "Actual data")
plt.plot(X_scaled, sv_regressor_scaled.predict(X_scaled), label = "Regressor line")
plt.legend()
plt.show()