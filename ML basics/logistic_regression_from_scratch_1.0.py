# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:41:30 2019

@author: chethan.k
"""

#importing libraries
#import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, confusion_matrix, roc_auc_score
#import matplotlib.pyplot as plt

#sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#converting the predicted outputs to classes
def classify(predictions):
    pred_classes = []
    for val in predictions:
        value = 1 if val >= .5 else 0
        pred_classes.append(float(value))
    return pred_classes

#predicting the output
def predict(data, weights):
    product = np.dot(data, weights)
    return sigmoid(product)

#cost function
# we use Cost = -sum(target*log(predictions) + (1-targget)*log(1-predictions) ) / len(target)
def cost_func(data, weights, target):
    N = len(data[0])
    pred = predict(data, weights)
    cost1 = (target * np.log(pred))
    cost2 = (np.ones(target.shape) - target) * np.log(np.ones(len(pred)) - pred)
    cost = -(cost1 + cost2)
    cost = cost.sum() / N
    return cost

#using gradient descent to update weights
#gradient = np.dot(X.T, (h - y)) / y.shape[0]
def gradient_descent(data, weights, target, learning_rate):
    N = len(data[0])
    pred = predict(data, weights)
    gradient = np.dot(data.T, (pred - target.reshape(-1,1)))
    gradient /= N
    gradient *= learning_rate
    weights -= gradient
    return weights

#training function for our model
def train(data, weights, target, learning_rate, epochs):
    cost_history = []
    for i in range(epochs):
        weights = gradient_descent(data, weights, target, learning_rate)
        cost = cost_func(data, weights, target)
        cost_history.append(cost)
        if (i % 100 == 0):
            print ("iter: ",str(i) , " cost: ",str(cost))
    return weights

#importing data and initialising variables
iris = load_iris()
X = iris.data[:100,:]
y = iris.target[:100]

#splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#training our model
weight = np.zeros([4,1])
weights = train(X_train, weight, y_train, 0.0001, 1001)
pred = predict(X_test, weights)
y_pred = classify(pred)

#accuracy function for the model
def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

#model scores
print("Score: ",accuracy(y_pred, y_test))
print("r2_score: ",r2_score(y_test, y_pred))
print("roc_auc_score: ",roc_auc_score(y_test, y_pred))
print("Confusion_matrix: ",confusion_matrix(y_test, y_pred))
