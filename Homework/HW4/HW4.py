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

#data =datasets.fetch_olivetti_faces(data_home='http://wayback.archive.org/web/*/http://www.uk.research.att.com/facedatabase.html')

dataset = fetch_olivetti_faces()
X = dataset.data
y = dataset.target
im = dataset.images

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
random_pca=pca_r.fit(X)
print(pca_r.explained_variance_ratio_)

for i in range(0,20):
    print "% of variance explained from component",i+1,"is: ", pca_r.explained_variance_ratio_[i]

######################################
X3 = data.data
y3 = data.target
im3 = data.images

pca3 = PCA(n_components = 2)
X_r3 = pca3.fit(X).transform(X)

ratios = pca3.explained_variance_ratio_

for i in range(0,2):
    print "% of variance explained from component",i+1,"is: ", pca3.explained_variance_ratio_[i]
    
comp_id = range(1,401)
fig = plt.figure(figsize=(8,5))
plt.plot(comp_id, ratios, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')

########################################

from sklearn.cluster import KMeans

dataset = fetch_olivetti_faces()
X = dataset.data
y = dataset.target
im = dataset.images

X_r = pca.fit(X).transform(X)

km = KMeans(n_clusters=10, init='random', n_init=10 , max_iter = 300, random_state=1)
km = KMeans(n_clusters=10, init='random', n_init=1 , max_iter = 1, random_state=1)

def do_kmeans(km, data):
    km.fit(data)
    centroids = km.cluster_centers_
    print "centroids:", centroids
    y = km.predict(data)
    
    
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    for t,marker,c in zip(xrange(10),">ox+>oxt>x","rgbycmykbg") :
        ax.scatter(data[y == t,0],
                   data[y == t,1],
                   marker=marker,
                   c=c)

    ax.scatter(centroids[:,0],centroids[:,1],marker = 's',c='r')

do_kmeans(km,X_r)

from sklearn import metrics
from sklearn.metrics import pairwise_distances

km.fit(X)
labels = km.labels_
metrics.silhouette_score(X, labels, metric='euclidean')

for k in xrange(2,11):
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(X)
    labels = km.labels_
    print k, metrics.silhouette_score(X, labels, metric='euclidean')







