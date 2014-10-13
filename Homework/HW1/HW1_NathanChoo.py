# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 20:11:00 2014

@author: nathanchoo
"""

print "hi"

import json    
from datetime import datetime

json_data = open("ga_hw_logins.json")
data = json.load(json_data)

time_list = []
for i in data:
    time_list.append(datetime.strptime(i, '%Y-%m-%d %H:%M:%S'))

for item in time_list:
    print item
    

