# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:50:25 2019

@author: chethan.k
"""

class Employee:
    
    raise_amount = 1.04
    #initialising the values
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
    
    @property   #@property defiines the function to be used as a property 
    def fullname(self):
        return "{} {}".format(self.first, self.last)
    
    @property   #@property defiines the function to be used as a property
    def email(self):
        return "{}.{}@email.com".format(self.first, self.last)
    
    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)