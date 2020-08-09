# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:53:43 2019

@author: chethan.k
"""

#importing libraries
#import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#importing the data
data = pd.read_csv("Salary_Data.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

#splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#initializing the decision tree regressor
dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(X_train, y_train)

#predicting the output
y_pred = dt_regressor.predict(X_test)

#printing model accuracy by r2_score
print("r2_score: ",r2_score(y_test, y_pred))