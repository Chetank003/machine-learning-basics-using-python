# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:06:50 2019

@author: chethan.k
"""
def Luhn_check():
    while True:
        
        card_number = int(input("Enter you valid 16_digit card number: "))
        #card_number = int(card_number)
        if(type(card_number) != int):
            raise TypeError("Invalid input, please enter only numeric values")
        if(len(str(card_number)) != 16):
            raise ValueError("Please enter valid '16-digit' card number")
            
        check_value = check(card_number)
        
        if(check_value):
            print("This is a valid card number")
        else:
            print("This is not a valid card number")
            
        choice = input("Would you like to check again? (Y/N): ")
        if(choice.lower()[0] == 'n'):
            break
        
def check(card_number):
    number = []
    for val in str(card_number):
        number.append(int(val))
    
    for i in range(0, len(number), 2):
        number[i] *= 2
        if(number[i] > 9):
            number[i] = (number[i] // 10) + (number[i] % 10)
    
    if(sum(number) % 10 == 0):
        return True
    else:
        return False
    
if __name__ == '__main__':
    Luhn_check()