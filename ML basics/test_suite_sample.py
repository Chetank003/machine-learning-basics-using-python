# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:19:18 2019

@author: chethan.k
"""
#importing libraries and modules
import unittest
import test_calculator
import test_employee

#initialise test loader and test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

#adding test cases to the test suite
suite.addTests(loader.loadTestsFromModule(test_calculator))
suite.addTests(loader.loadTestsFromModule(test_employee))

#initializing the test runner and runnig the test suite
runner = unittest.TextTestRunner()
runner.run(suite)