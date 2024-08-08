#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import svm


#read dataset
data=pd.read_csv("./nba2021.csv")

#Encoding the categorical values
labelen = LabelEncoder()
data['Pos']=labelen.fit_transform(data['Pos'])
data['Tm']=labelen.fit_transform(data['Tm'])

#In each position eliminate records whose duration belongs to bottom 10% of data
percent=10
filtered_dataframes = []

for pos in range(5):
    data_pos = data[data["Pos"] == pos]
    arr = data_pos["MP"]
    arr = np.array(arr)
    percent_value = np.percentile(arr, percent)
    data_filtered = data_pos[data_pos["MP"] > percent_value]
    filtered_dataframes.append(data_filtered)
    new_df2 = pd.concat(filtered_dataframes)

# Dropping the features that are redundant and unwanted
new_df2 = new_df2.drop(columns=["Player", "Tm", "FG", "FGA", "ORB", "DRB", "2P", "2PA", "3P", "3PA","FT", "FTA"])

# Filling the empty values
new_df2.fillna(0)

# Task - 1
# Creating objects for input and output features
X = new_df2.drop('Pos', axis=1)
Y = new_df2.Pos

#spilting of data set into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42) 

#Create a svm Classifier
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

#predicting the values
Y_pred = model.predict(X_test)

training_score = model.score(X_train, Y_train)
test_score = model.score(X_test, Y_test)

#printing accurancy
print("Train set score: {:.3f}".format(model.score(X_train, Y_train)))
print("Test set score: {:.3f}".format(model.score(X_test, Y_test)))
print("Accuracy: {:.3f}".format(metrics.accuracy_score(Y_test, Y_pred)))

#printing confusion matrix
print("Confusion matrix:")
print(pd.crosstab(Y_test, Y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

#cross-validation and its average
sf=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
scores = cross_val_score(model, X, Y, cv=sf)
print("Accuracy in each fold is: {}".format(scores))
print("Average accuracy across all the folds: {:.2f}".format(scores.mean()))


# In[ ]:


s

