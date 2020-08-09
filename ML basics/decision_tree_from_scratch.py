# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:43:44 2019

@author: chethan.k
"""
#importing libraries
from sklearn.datasets import load_iris
import random
import numpy as np

#importing the data
iris = load_iris()
data = iris.data
target = iris.target_names[iris.target]

#defing the train_test_split method
def train_test_split(data, target, test_size):
    random.seed(0)
    if(type(test_size) == float):
        test_size *= len(data)
        test_size = round(test_size)
    indices = []
    for i in range(0, len(data)):
        indices.append(i)
    random_indices = random.sample(indices, test_size)
    test_data = data[random_indices]
    train_data = data[[x for x in indices if x in random_indices]]
    test_target = data[random_indices]
    train_target = data[[x for x in indices if x in random_indices]]
    return  train_data, test_data, train_target, test_target

def check_purity(data, target):
    unique_classes = np.unique(target)
    if(len(unique_classes) == 1):
        return True
    else:
        return False
    
