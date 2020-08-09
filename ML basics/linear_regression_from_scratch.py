# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:00:25 2019

@author: chethan.k
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
#creating dummy data
X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
'''
data = pd.read_csv("Salary_Data.csv")
X = data.iloc[:,0].values
y = data.iloc[:,-1].values

#visualizing the data
plt.scatter(X, y, 50, np.random.rand(np.size(X)))
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Plot of values for LR")
plt.show()

'''
for a simple linear regression y = bo + b1X + E
here bo is the intercept and b1 is the slope of the regression line and E is the error
so we need to find bo and b1
'''
#means of X and y
X_mean = np.mean(X)
y_mean = np.mean(y)

#cross deviation and deviation along X
cross_deviation_XY = np.sum(X * y) - np.size(X)*X_mean*y_mean
deviation_XX = np.sum(X * X) - np.size(X)*X_mean*X_mean

#calculating the coeficients bo and b1
b1 = cross_deviation_XY / deviation_XX
bo = y_mean - b1 * X_mean
print("bo: ",bo)
print("b1: ",b1)

#predicting the values of y using y = bo + b1X and plotting the regression line
y_pred = bo + (b1 * X)
plt.scatter(X, y)
plt.plot(X, y_pred, color = "green")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Plot of regressor line against the actual values of y")
plt.show()

#linear regression using sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X = X.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_sklearn = linear_regressor.predict(X_test)

#plotting the model
plt.scatter(X, y, 50)
plt.plot(X_test, y_pred_sklearn, color = "green")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Plot of regressor line from sklearn against the actual values of y")
plt.show()

#evaluating the model
'''
for model evaluation we will use r2_score and r2_score = 1 - (RSS/TSS). RSS(Residual sum of squares), TSS(Total sum of squares)
RSS = sum(yi - yfitted)^2         TSS = sum(yi-ymean)^2
'''
#finding RSS
RSS = 0
difference_rss = y - y_pred
for val in difference_rss:
    RSS += val*val
print("variance in our linear model: ", RSS)

#finding TSS
y_mean_array = np.full(np.size(y), y_mean)
TSS = 0
difference_tss = y - y_mean_array
for val in difference_tss:
    TSS += val*val
print("Variance in the target variable: ", TSS)

#Finding r2_score and comparing it with r2_score of sklearn.metrics
r2_score_calculated = 1 - (RSS/TSS)
from sklearn.metrics import r2_score
r2_score_sklearn = r2_score(y, y_pred)
print("r2_score_calculated: ",r2_score_calculated)
print("r2_score_sklearn: ",r2_score_sklearn)