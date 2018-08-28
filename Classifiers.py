#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import amax
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import os
import time

# ------------ DEFINIÇÃO DOS MÉTODOS NECESSÁRIOS ------------- #
def crossValKNN(range_k, X_data, y_target, algoritmo):
	scores = []

	inic = time.time()

	for k in range_k:
		neigh = KNeighborsClassifier(k, algorithm=algoritmo)
		neigh.fit(X_data, y_target)
		scores.append(cross_val_score(neigh, X_data, y_target, cv=5).mean())

	fim = time.time()

	return scores, (fim - inic)

def splitKNN(range_k, X_data, y_target, algoritmo):
	scores = []

	inic = time.time()

	for k in range_k:
		neigh = KNeighborsClassifier(k, algorithm=algoritmo)
		X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.4, random_state=0)
		neigh.fit(X_train, y_train)
		scores.append(accuracy_score(y_test, neigh.predict(X_test)))

	fim = time.time()

	return scores, (fim - inic)

def printAccurancyVsK(range_k, scoresSplit, scoresCross, title):
	plt.subplot(121)
	plt.plot(range_k, scoresSplit)
	plt.xlabel('K')
	plt.ylabel('Acuracia')
	plt.title('Split 60/40')

	plt.suptitle(title, fontsize=16)
	
	plt.subplot(122)
	plt.plot(range_k, scoresCross)
	plt.xlabel('K')
	plt.ylabel('Acuracia')
	plt.title('Cross Validation')
	
	plt.show()
	plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
		
def representacaoDigits(X, Y, algoritmo):
	os.system("python digits.py %s %s" % (X, Y))

	X_data_digits, y_target_digits = datasets.load_svmlight_file('features.txt')
	
	splitKNNScores, timeSplit = splitKNN(range_k, X_data_digits, y_target_digits, algoritmo)
	crossValScores, timeCross = crossValKNN(range_k, X_data_digits, y_target_digits, algoritmo)
	
	printAccurancyVsK(range_k, splitKNNScores, crossValScores, 'Resultados para X=%s e Y=%s' % (X, Y))

	os.system("rm features.txt")

	return timeSplit, timeCross, max(splitKNNScores), max(crossValScores)



# ------------------------- EXECUÇÃO DAS ATIVIDADES PARA O IRIS DATASET -------------------------- #
iris = datasets.load_iris()

X = iris.data
y = iris.target

range_k = range(1, 30)
splitKNNScores = []
crossValScores = []

splitKNNScores, _ = splitKNN(range_k, X, y, 'brute')
crossValScores, _ = crossValKNN(range_k, X, y, 'brute')

printAccurancyVsK(range_k, splitKNNScores, crossValScores, 'Iris Dataset Resultados')




# -------------- EXECUÇÃO DO DATASET DIGITS ----------------------- #
representacaoDigits('20', '10', 'brute') # 92.8% - Split        92.6% - Cross Validation
representacaoDigits('3', '7', 'brute')   # 78.4% - Split        80.3% - Cross Validation
representacaoDigits('7', '5', 'brute')   # 85.0% - Split        85.4% - Cross Validation
representacaoDigits('10', '12', 'brute') # 93.2% - Split        93.0% - Cross Validation
representacaoDigits('5', '5', 'brute')   # 82.7% - Split        82.7% - Cross Validation
representacaoDigits('4', '8', 'brute')   # 85.8% - Split        85.8% - Cross Validation

# --------------- MEDIDA DE EFICÁCIA X TEMPO ----------------------- #



tSplit, tCross, maxSplit, maxCross = representacaoDigits('10', '12', 'brute')
