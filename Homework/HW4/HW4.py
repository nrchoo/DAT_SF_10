# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 22:25:25 2014

@author: nathanchoo
"""
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd
%matplotlib inline

# scikit-learn algorithms

from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.datasets import fetch_olivetti_faces


data =datasets.fetch_olivetti_faces(data_home='http://wayback.archive.org/web/*/http://www.uk.research.att.com/facedatabase.html')

X = data.data
y = data.target
im = data.images

pca = PCA()
X_r = pca.fit(X).transform(X)

ratios = pca.explained_variance_ratio_

for i in range(0,20):
    print "% of variance explained from component",i+1,"is: ", pca.explained_variance_ratio_[i]
    
comp_id = range(1,401)
fig = plt.figure(figsize=(8,5))
plt.plot(comp_id, ratios, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
###################################
'''
from numpy.random import RandomState
rng = RandomState(0)
data2 = fetch_olivetti_faces(shuffle=True, random_state=rng)

X2 = data2.data
y2 = data2.target
im2 = data2.images

pca2 = PCA()
X_r2 = pca2.fit(X2).transform(X2)

ratios2 = pca2.explained_variance_ratio_

for i in range(0,20):
    print "% of variance explained from component",i+1,"is: ", pca2.explained_variance_ratio_[i]

comp_id = range(1,401)
fig = plt.figure(figsize=(8,5))
plt.plot(comp_id, ratios2, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
'''
######################################
from sklearn.decomposition import RandomizedPCA

pca_r = RandomizedPCA()
pca_r.fit(X)
print(pca_r.explained_variance_ratio_)











