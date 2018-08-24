#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import datasets
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Load do dataset
iris = datasets.load_iris()

# Extração de 4 características
X = iris.data[:, :4]
y = iris.target

# Para a divisão do database em porcentagens de treinamento e de teste. EXECUTAR 30x.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

neigh = NearestNeighbors(2, 0.4)
# neigh.fit(X)
# train = neigh.kneighbors(X, return_distance=False)

kf = KFold(n_splits=5)
for train, test in kf.split(X):
	X_train, X_test = X[train], X[test]
   	y_train, y_test = y[train], y[test]
    
neigh.fit(X_train)
print(neigh.kneighbors(X_test, return_distance=False))

# print(cv_results)

