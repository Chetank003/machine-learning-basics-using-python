# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:41:47 2019

@author: chethan.k
"""

def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if(y == 0):
        raise ValueError("Cannot divide by zero!!")
    else:
        return x / y
    