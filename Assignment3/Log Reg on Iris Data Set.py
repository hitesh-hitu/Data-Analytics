#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd
from sklearn import decomposition
from sklearn import datasets
# Importing the dataset
dataset = pd.read_csv('C:/Users/Hitesh Kumar/Videos/iris.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
#print(X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#print(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0,solver='lbfgs',multi_class='auto')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

print("Mean squared error: %.2f" % np.mean((classifier.predict(X_test) - y_test) ** 2))

print('Variance score: %.2f' % classifier.score(X_test, y_test))



# In[ ]:




