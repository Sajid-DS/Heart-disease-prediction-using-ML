# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 21:32:27 2021

@author: -
"""
#Importing Important Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

#Loading Dataset
data = pd.read_csv(r'C:\Users\hp\OneDrive\Desktop\Project_to_upload\heart.csv')

#Performing basic EDA
print(data.describe())
print(data.info())

corelation = data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corelation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':9},cmap='Reds')


X = data.iloc[:,:-1]
y = data.iloc[:,-1:]

#Dividing data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

#creating LogisticRegression model
model = LogisticRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_test)


print('Performance on Testing set ->',accuracy_score(y_test,prediction))

with open('model.pkl','wb') as file:
    pickle.dump(model, file)
