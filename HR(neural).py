# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:32:00 2020

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#importing dataset
neural= pd.read_csv("C:\\Users\\User\\Downloads\\HRDataset.csv")

neural.columns
neural = neural.iloc[:,2:]
neural.isnull().sum()


# data preprocessing
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
neural.iloc[:, 7] = labelencoder_X.fit_transform(neural.iloc[:, 7])
neural.iloc[:, 8] = labelencoder_X.fit_transform(neural.iloc[:, 8])
neural.iloc[:, 9] = labelencoder_X.fit_transform(neural.iloc[:, 9])
#splitting data in train and test
train,test = train_test_split(neural,test_size = 0.3,random_state=42)
trainX = train.drop(["PerformanceScore"],axis=1)
trainY = train["PerformanceScore"]
testX = test.drop(["PerformanceScore"],axis=1)
testY = test["PerformanceScore"]

#To standardized the data using from sklearn.preprocessing import StandardScaler package
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(trainX)
trainX = scalar.transform(trainX)

#MLPClassifier stands for Multi-layer Perceptron classifier, It connects to a neural network 
from sklearn.neural_network import MLPClassifier
mpl = MLPClassifier(hidden_layer_sizes=(30,30))
mpl.fit(trainX,trainY)
predection_train = mpl.predict(trainX)
predection_test = mpl.predict(testX)
#To evaluate the accuracy of a classification using from sklearn.metrics import confusion_matrix package
from sklearn.metrics import confusion_matrix
print(confusion_matrix(testY, predection_test))
np.mean(testY==predection_test) #accuracy= 0.83
np.mean(trainY==predection_train) #accuracy= 1.0

