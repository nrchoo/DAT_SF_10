# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 20:11:00 2014

@author: nathanchoo
"""

import json, sqlite3
from datetime import datetime
import pandas as pd

json_data = open("ga_hw_logins.json")
data = json.load(json_data)

time_list = []
for i in data:
    time_list.append(datetime.strptime(i, '%Y-%m-%d %H:%M:%S'))

'''for item in time_list:
    print item'''
    
#Database portion
   
conn = sqlite3.connect('example.db')
c = conn.cursor()
c.execute('create table logins (date text)')
conn.commit()

for i in time_list:
    t = (i,)
    c.execute('INSERT INTO logins VALUES (?)', t)

'''c.execute('select * from logins')
print c.fetchall()'''


#Most logins by day
c.execute('''select strftime("%d",date) yr_mon, 
count(*) num_dates from logins group by yr_mon''')

group_date = c.fetchall()
group_date_pd = pd.DataFrame(group_date)
max_login_day = group_date_pd[group_date_pd[1] == group_date_pd[1].max()]
max_login_day.columns = ['day','count']


#Most logins by hour
c.execute('''select strftime("%H",date) hour, 
count(*) num_dates from logins group by hour''')
group_hour = c.fetchall()
group_hour_pd = pd.DataFrame(group_hour)
max_login_hour = group_hour_pd[group_hour_pd[1] == group_hour_pd[1].max()]
max_login_hour.columns = ['hour', 'count']

#Report

print "Most logins by day:", max_login_day
print "Most logins by hour:", max_login_hour

#Clean up

c.execute('drop table example.logins')
conn.close()