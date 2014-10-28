# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 18:49:11 2014

@author: nathanchoo
"""

import pandas as pd
from pandas.tools.plotting import scatter_matrix
import statsmodels.api as sm

data = pd.read_csv('student_logins.csv')

#Convert all times into calculatable format
data['session_end_time'] = pd.to_datetime(data['session_end_time'])
data['session_start_time'] = pd.to_datetime(data['session_start_time'])
data['student_account_created'] = pd.to_datetime(data['student_account_created'])
data['Duration'] = data['session_end_time'] - data['session_start_time']
data['Duration'] = data['Duration'].map(lambda x: x.astype('float64')/(1e9*60))

#Create Account Age variable
MaxCreated = pd.to_datetime(max(data['session_end_time']))
data['AccountAge'] = MaxCreated - data['student_account_created']
data['AccountAge'] = data['AccountAge'].map(lambda x: x.astype('float64')/(1e9*60*60*24))

#Use dummies to created class boolean
class_id = data['class_id']
class_dummies = pd.get_dummies(class_id)
data_m = pd.merge(data,class_dummies,left_index=True,right_index=True,how='left')

#Remove outliers 0 logins and max logins
data_m = data_m[(data_m['Duration'] < 225) & (data_m['Duration'] > 0)]

#Create scatter plot on all data
scatter_1 = scatter_matrix(data, figsize = (15,15))

'''
Possible relationships:
1. Duration is a function of student_previous_logins_total
2. Duration is a function of student_previous_class_logins
3. Duration is a function of account age
'''

#Test to see if duration is a function of classes taken
data_m2 = data_m[['Duration','a','c','e','g','m']]
scatter_2 = scatter_matrix(data_m2, figsize = (10,10))

#Reduce dataset to look at fewer features
data_m3 = data_m[['problems_completed','AccountAge','student_previous_logins_total','student_previous_class_logins','Duration']]
scatter_3 = scatter_matrix(data_m3, figsize = (10,10))


x = data_m[['problems_completed','AccountAge','student_previous_logins_total','student_previous_class_logins','a','c','e','g','m']].values
y = data_m['Duration'].values
X = sm.add_constant(x, prepend=True)
results = sm.OLS(y, X).fit()
results.summary()

#I selected the features above because after looking at the scatter plots , the graphs seemed to have shown some correlation between duration time


'''
I chose to look at the R squared value for this analysis as shown below. 
R-squared: 0.486. The conclusion for this particular model is that this dataset doesn't show any strong correlation between duration and features.
'''


