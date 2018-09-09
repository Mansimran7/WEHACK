# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 13:19:44 2018

@author: Mansimran Anand
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('DataSet.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 9].values

#aplying linear regression 
from sklearn.linear_model import LogisticRegression 
regressor = LogisticRegression()
regressor.fit(X,Y)

#predicted values 
y_pred = regressor.predict(X)

#finding mean squared error (by applying linear regression)
from sklearn.metrics import mean_squared_error
mean_squared_error(Y,y_pred)

#removing all those columns that are not important or let's just say have no importance on the output variable 

