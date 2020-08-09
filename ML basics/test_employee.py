# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:55:55 2019

@author: chethan.k
"""

import unittest
from employee import Employee

class TestEmployee(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("setupclass")
        
    @classmethod
    def tearDownClass(cls):
        print("teardownclass")
    
    #setUp() method runs before each test method
    def setUp(self):
        print("Setup")
        #creating employee objects for each test methods
        self.emp1 = Employee("chethan", "k" , 50000)
        self.emp2 = Employee("vinay", "kumar" , 60000)
        
    #tearDown() method runs after each test method
    def tearDown(self):
        print("Teardown")
        
    def test_email(self):
        print("Testing emial")
        self.assertEqual(self.emp1.email, "chethan.k@email.com")
        self.assertEqual(self.emp2.email, "vinay.kumar@email.com")
        
        self.emp1.first = "chetan"
        self.emp2.first, self.emp2.last = self.emp2.last, self.emp2.first
        
        self.assertEqual(self.emp1.email, "chetan.k@email.com")
        self.assertEqual(self.emp2.email, "kumar.vinay@email.com")
        
    def test_fullname(self):
        print("Testing fullname")
        self.assertEqual(self.emp1.fullname, "chethan k")
        self.assertEqual(self.emp2.fullname, "vinay kumar")
        
        self.emp1.first = "chetan"
        self.emp2.first, self.emp2.last = self.emp2.last, self.emp2.first
        
        self.assertEqual(self.emp1.fullname, "chetan k")
        self.assertEqual(self.emp2.fullname, "kumar vinay")
        
    def test_applyraise(self):
        print("Testing appraisal")
        self.emp1.apply_raise()
        self.emp2.apply_raise()
        
        self.assertEqual(self.emp1.pay, 52000)
        self.assertEqual(self.emp2.pay, 62400)
        
if __name__ == '__main__':  #if this program is run by main method
    unittest.main()