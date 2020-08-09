# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:14:07 2019

@author: chethan.k
"""

#!/usr/bin/python
#import pymysql
#conn=pymysql.connect(host="localhost", user="root", passwd="Attra#11", db="atm_dashboard")
#c=conn.cursor()
#csv_data=csv.reader(open("dump.txt"))
#for row in scv_data:
  #  c.execute("INSERT INTO atm_dashboard.atm_device_status()values()%tuple(row)")
   # c.execute("SELECT * FROM atm_dashboard.atm_device_status")
    #print(row)
#conn.commit()
#conn.close()

################ TO Display the ATM
#import pymysql
#def doQuery(conn):
#    cur=conn.cursor()
#    cur.execute("CREATE TABLE IF NOT EXISTS atm_dashboard.ATM_STATUS(ATM_ID varchar(50) PRIMARY KEY NOT NULL, ATM_HEALTH varchar(50),ATM_ADDRESS varchar(100));")
#    cur.execute("LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 5.7/Uploads/dump.txt' INTO TABLE atm_dashboard.ATM_STATUS FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'")
#    cur.execute("SELECT * FROM atm_dashboard.ATM_STATUS")
#    for row in cur.fetchall():
#        print(row)
#myConnection=pymysql.connect(host="localhost", user="root", passwd="Attra#11", db="atm_dashboard")
#doQuery(myConnection)
#myConnection.commit()
#myConnection.close()



################ TO Display the ATM STATUS REPORT 1 ######################
import pymysql
#import geopy
from geopy import Nominatim
nom=Nominatim()
def doQuery(conn):
    cur=conn.cursor()
        ###### CREATE VIEW #######
    #cur.execute("CREATE VIEW atm_dashboard.ATM_STATUS AS SELECT ADS_DEVICE_ID AS ATM_DEVICE_ID, ADS_HEALTH AS ATM_STATUS, concat(ALO_ADDRESS_1, ALO_ADDRESS_2, ALO_ADDRESS_3, ALO_ADDRESS_4, ALO_ADDRESS_5, ALO_COUNTRY_CODE) AS ATM_ADRESS FROM ATM_DEVICE_STATUS, ATM_LOCATIONS WHERE ATM_DEVICE_STATUS.ADS_LAST_UPDATE_TS=ATM_LOCATIONS.ALO_LAST_UPDATE_TS;")
	#OR
    #cur.execute("CREATE VIEW atm_dashboard.ATM_STATUS AS SELECT ADS_DEVICE_ID AS ATM_ID, ADS_HEALTH AS ATM_STATUS, Concat(ALO_ADDRESS_1, ALO_ADDRESS_2, ALO_ADDRESS_3, ALO_ADDRESS_4, ALO_ADDRESS_5, ALO_COUNTRY_CODE) AS ATM_LOCATION, ALO_LOCATION_ID AS ATM_REGION FROM ATM_DEVICE_STATUS, ATM_LOCATIONS WHERE ADS_LAST_UPDATE_TS = (SELECT ADS_LAST_UPDATE_TS FROM ATM_DEVICE_STATUS JOIN ATM_DEVICE_TOTALS ON ATM_DEVICE_TOTALS.ATO_AST_ID = ATM_DEVICE_STATUS.ADS_AST_ID);")
    
        ###### CREATE ATM_STATUS_DASHBOARD table ######
    cur.execute("CREATE TABLE IF NOT EXISTS ATM_STATUS_DASHBOARD(ATM_ID varchar(50) PRIMARY KEY NOT NULL, ATM_STATUS varchar(50),ATM_LOCATION varchar(100), ATM_REGION varchar(50),ATM_LONGITUDE float,ATM_LATITUDE float)")
    cur.execute("SELECT * FROM ATM_STATUS;")
    for row in cur.fetchall():
	    ###### Script to generate longitude and latitude  ######
        location=row[4]
        n=nom.geocode(location)
        lat=n.latitude
        long=n.longitude
           ###### Insert values into dashboard table ATM_STATUS_DASHBOARD ######
        cur.execute("""INSERT INTO ATM_STATUS_DASHBOARD(ATM_ID, ATM_STATUS, ATM_LOCATION, ATM_REGION,ATM_LONGITUDE,ATM_LATITUDE)values(%s,%s,%s,%s,%s,%s)""",(row[0],row[0],row[2],row[3],long,lat))
        
    conn.commit()
    cur.close()
        
conn=pymysql.connect(host="localhost", user="root", passwd="root", db="authentic")
doQuery(conn)
conn.close()