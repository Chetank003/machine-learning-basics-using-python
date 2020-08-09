# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:00:00 2019

@author: chethan.k
"""

#importing librares
#import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#predicting the outputs
def predict(experience, weight, bias):
    return experience*weight + bias

#using MSE as cost function
def cost_func(experience, weight, bias, salary):
    mse = 0
    N = len(experience)
    for i in range(N):
        mse += (salary[i] - (experience[i]*weight + bias)) ** 2
    return mse/N

#using gradient descent approach to update weight and bias
def gradient_descent(experience, weight, bias, salary, learning_rate):
    weight_derivative = 0
    bias_derivate = 0
    N = len(experience)
    for i in range(N):
        weight_derivative += -2*experience[i] * (salary[i] - (experience[i] * weight + bias))
        bias_derivate += -2 * (salary[i] - (experience[i] * weight + bias))
    weight -= (weight_derivative/N) * learning_rate
    bias -= (bias_derivate/N) * learning_rate
    return weight, bias

#function for training the model
def train_model(experience, weight, bias, salary, learning_rate, epochs):
    cost_history = []
    for i in range(epochs):
        weight, bias = gradient_descent(experience, weight, bias, salary, learning_rate)
        cost = cost_func(experience, weight, bias, salary)
        cost_history.append(cost)
        if(i == epochs - 1):
            print("iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2f}".format(i, weight, bias, cost_history[i]))
    
    return weight, bias

#importing the dataset and initialising the variables
data = pd.read_csv("Salary_Data.csv")
X = data.iloc[:, 0].values
y = data.iloc[:, -1].values

#splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#training the model
weight, bias = train_model(X_train, 0, 0, y_train, 0.02, 1001)
y_pred = predict(X_test, weight, bias)
print("r2_score: ",r2_score(y_test, y_pred))

#visualising the trained model
plt.scatter(X, y, 50, label = 'Data')
plt.plot(X_test, y_pred, label = 'Model_prediction')
plt.legend()
plt.show()