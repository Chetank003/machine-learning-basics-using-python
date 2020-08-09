# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:45:43 2019

@author: chethan.k
"""

import unittest
import calculator

class TestCalc(unittest.TestCase):
    
    def test_add(self):
        self.assertEqual(calculator.add(10, 5), 15)
        self.assertEqual(calculator.add(10, -5), 5)
        self.assertEqual(calculator.add(-10, -5), -15)
    
    def test_subtract(self):
        self.assertEqual(calculator.subtract(10, 5), 5)
        self.assertEqual(calculator.subtract(10, -5), 15)
        self.assertEqual(calculator.subtract(-10, -5), -5)
        
    def test_multiply(self):
        self.assertEqual(calculator.multiply(10, 5), 50)
        self.assertEqual(calculator.multiply(10, -5), -50)
        self.assertEqual(calculator.multiply(-10, -5), 50)
        
    def test_divide(self):
        self.assertEqual(calculator.divide(10, 5), 2)
        self.assertEqual(calculator.divide(10, -5), -2)
        self.assertEqual(calculator.divide(-10, -5), 2)
        self.assertRaises(ValueError, calculator.divide, 10, 0)
        #or we can use context manager like:
        '''
        with self.assertRaises(ValueError):
            calculator.divide(10, 0)
        '''
        
    @unittest.SkipTest
    def test_skipped(self):
        print("This will be skipped")
if __name__ == '__main__':
    unittest.main()