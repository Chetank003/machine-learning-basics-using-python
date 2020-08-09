# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:28:25 2019

@author: chethan.k
"""

import mysql.connector
from mysql.connector import Error

try:
    connection = mysql.connector.connect(host = "localhost",database = "authentic",user = "root",password = "root")
    cursor = connection.cursor()
    query = "select * from atm_device_totals;"
    cursor.execute(query)
    data = cursor.fetchall()
    
    for row in data:
        print(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],"\n")
    cursor.close()
    
except Error as e:
    print("Error in script: ",e)

finally:
    if(connection.is_connected()):
        connection.close()