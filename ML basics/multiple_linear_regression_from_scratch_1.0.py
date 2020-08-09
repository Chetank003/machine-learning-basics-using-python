# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:29:23 2019

@author: chethan.k
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#predicting the outputs
def predict(features, weights):
    return np.dot(features, weights)

#Normalizing the faetures
'''
since our data has vast deviation in values we need to perform feature scaling on the data 
so we normalise the data by doing:
    For each feature column {
    #1 Subtract the mean of the column (mean normalization)
    #2 Divide by the range of the column (feature scaling)
}
'''
def normalize(features):
    features = (features - features.mean()) / (features.max() - features.min())
    return features

#using MSE as the cost function
#we divide MSE by 2 to make calculations of derivatives simpler
def cost_func(features, weights, targets):
    pred = predict(features, weights)
    squares = (targets - pred) ** 2
    cost = np.sum(squares) / (2 * len(features))
    return cost

#using gradient descent to update weights
'''
for our data we donot need bias because 
the business logic says there can be no profit if no money is spent for the respective departments
'''
def gradient_descent(features, weights, targets, learning_rate):
    pred = predict(features, weights)
    f1 = features.iloc[:, 0].values
    f2 = features.iloc[:, 1].values
    f3 = features.iloc[:, 2].values
    f4 = features.iloc[:, 3].values
    f5 = features.iloc[:, 4].values
    
    w1 = -f1 * (targets - pred)
    w2 = -f2 * (targets - pred)
    w3 = -f3 * (targets - pred)
    w4 = -f4 * (targets - pred)
    w5 = -f5 * (targets - pred)
    
    weights[0][0] -= (np.mean(w1) * learning_rate)
    weights[1][0] -= (np.mean(w2) * learning_rate)
    weights[2][0] -= (np.mean(w3) * learning_rate)
    weights[3][0] -= (np.mean(w4) * learning_rate)
    weights[4][0] -= (np.mean(w5) * learning_rate)
    
    return weights

#training function
def train_model(features, weights, targets, learning_rate, epochs):
    cost_history = []
    features = normalize(features)
    for i in range(epochs):
        weights = gradient_descent(features, weights, targets, learning_rate)
        cost = cost_func(features, weights, targets)
        if(i % 100 == 0):
            print("iter={:d}    weight={:.2f}    cost={}".format(i, weights, cost_history[i]))
        cost_history.append(cost)
    return weights

#importing the dataset
data = pd.read_csv("50_Startups.csv")
X = data.iloc[:, :4]
y = data.iloc[:, -1]
X = pd.get_dummies(X)
X = X.iloc[:,:-1]

#splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training the model
weights  = np.zeros([len(X_train.columns),1])
weights = train_model(X_train, weights, y_train, 0.0001, 1000)
y_pred = predict(X_test, weights)
print("r2_score: ",r2_score(y_test, y_pred))