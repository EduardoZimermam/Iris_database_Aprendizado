#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Load do dataset
iris = datasets.load_iris()

# Extração de 4 características com as 3 classes do Iris Dataset.
X = iris.data
y = iris.target

range_k = range(1, 30)
scores=[]

for k in range_k:
	neigh = KNeighborsClassifier(k)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
	neigh.fit(X_train, y_train)
	scores.append(accuracy_score(y_test, neigh.predict(X_test)))

plt.plot(range_k, scores)
plt.xlabel('K')
plt.ylabel('Acuracia')
plt.title('Usando 60% para treinamento e 40% para')
plt.show()

scores = []

for k in range_k:
	neigh = KNeighborsClassifier(k)
	neigh.fit(X, y)
	scores.append(cross_val_score(neigh, X, y, cv=5).mean())

plt.plot(range_k, scores)
plt.xlabel('K')
plt.ylabel('Acuracia')
plt.title('Usando 5 folds')
plt.show()