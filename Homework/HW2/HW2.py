# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 16:17:49 2014

@author: nchoo
"""
import pandas as pd
import csv
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


#data = pd.read_csv('Desktop\iris.csv')
#data = pd.DataFrame(data)
#X = np.array(data.ix[:,0:4])
#y = np.array(data.ix[:,-1])

foo = csv.reader(open('/Users/nathanchoo/Desktop/iris.csv'), delimiter = ',')
data = list(foo)
data.pop()
data = np.array(data)

X = data[:,:4]
y = data[:,-1]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=0)

knn = KNeighborsClassifier(n_neighbors=3)
knn. fit(X,y)

from sklearn.cross_validation import KFold
 
#generic cross validation function
def cross_validate(X, y, classifier, k_fold) :

    # derive a set of (random) training and testing indices
    k_fold_indices = KFold( len(X), n_folds=k_fold,
                           indices=True, shuffle=True,
                           random_state=0)

    k_score_total = 0
    # for each training and testing slices run the classifier, and score the results
    for train_slice, test_slice in k_fold_indices :

        model = classifier(X[ train_slice  ],
                         y[ train_slice  ])

        k_score = model.score(X[ test_slice ],
                              y[ test_slice ])

        k_score_total += k_score

    # return the average accuracy
    return k_score_total/k_fold
    
    
cross_validate(X,y,KNeighborsClassifier(3).fit, 5)

x_list = []
for i in range(1,151):
    x_list.append(cross_validate(X,y,KNeighborsClassifier(i).fit, 5))
    
for i in range(1,150):
   if x_list[i] == max(x_list):
       print "Optimum number of neighbors:", i +1, "with Score:", x_list[i]
      
    
        
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

xx = range(1,151)
plt.plot(xx,x_list)

#Bonus
y_list = []
for i in range(2,50):
    y_list.append(cross_validate(X,y,KNeighborsClassifier(11).fit, i))

yy = range(1,49)
plt.plot(yy,y_list)
print "Min score is:", min(y_list)
print "Max score is:", max(y_list)

#There is an optimal number of folds to use for cross validation, but it doesn't make much of a difference and it might hurt the model by overfitting.



    
    