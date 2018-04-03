#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 18:42:54 2018

@author: astaroth
"""

# Data Preprocessing

# Importing the libraries
import numpy as no
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values

# Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features= [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# importing the keras lib and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN

classifier = Sequential()

# Adding the input layer and the first input layer
classifier.add(Dense(input_dim=11,init='uniform',activation='relu',output_dim=6))

# Adding a second hidden layer
classifier.add(Dense(init='uniform',activation='relu',output_dim=6))

# Adding the output layer
classifier.add(Dense(init='uniform',activation='sigmoid',output_dim=1))

# compiling the ANN
classifier.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(x_train, y_train, batch_size=10, epochs=100)

#predicting the set results
y_pred = classifier.predict(x_test)

y_pred = (y_pred>0.5)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# cross-validation
def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(input_dim=11, init='uniform', activation='relu',output_dim=6))
    classifier.add(Dense(init='uniform', activation='relu',output_dim=6))
    classifier.add(Dense(init='uniform', activation='sigmoid',output_dim=1))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)

mean = accuracies.mean()
variance = accuracies.std()

print("Accuracy--- mean:"+str(mean)+("\nAccuracy--- variance:")+str(variance))

#tuning
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(input_dim=11, init='uniform', activation='relu',output_dim=6))
    classifier.add(Dense(init='uniform', activation='relu',output_dim=6))
    classifier.add(Dense(init='uniform', activation='sigmoid',output_dim=1))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring ='accuracy',
                           cv = 10)
grid_search = grid_search.fit(x_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

