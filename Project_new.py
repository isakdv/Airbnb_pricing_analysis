# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:58:49 2023

@author: Daria Isaak
"""
import os
import pandas as pd
import numpy as np
from datetime import  datetime, timedelta
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
print(tf.__version__)



df = pd.read_csv("AB_NYC_2019.csv")
df_up = df.copy() #we make a copy in case we want to modify our data and compare to the old version


######### Regression
######### Clean data

#check how many nan we have
df_up.isna().sum()
df_up.isnull().sum() #same


######### Modify our data


# working with last_review data -> assign the earliest date
# first, let's change str to date.format (it is easier to work with this particular format)
df_up['last_review'] = df_up['last_review'].astype('datetime64[ns]')
most_recent_date = df_up['last_review'].max() #find the most recent date
print(most_recent_date)

most_latest_date = df_up['last_review'].min() #find the earliest date
print(most_latest_date)

# fill nan with the earliest date (so we can avoid deleting 10k nan in last_review)
df_up['last_review'] = df_up['last_review'].fillna(most_latest_date) 

# working with review_per_month -> assign zero to nan
df_up['reviews_per_month'] = df_up['reviews_per_month'].fillna(0)

# delete zero prices
df_up['price'] = df_up['price'].replace(0, np.nan)

# Let's drop the rest
df_up.isna().sum()
df_up = df_up.dropna()
df_up.isna().sum()


######################################

# Now we delete info that we don't need (doesn't carry sensible information)
# and transform the rest of info into numerical values


del df_up['id'] #since id is a unique attribute - we don't need it
del df_up['host_id'] #since id is a unique attribute - we don't need it

# change description to the length of the description (i.e. maybe the longer - the better)
df_up['name_length']  = df_up['name'].str.len()
del df_up['name']

# we also surpass 'neighbourhood' and use lattitude and longitude instead (too many categorical variables)
del df_up['neighbourhood']

# tranform dates
# if it has the most recent date -> 0
# otherwise -> difference
df_up['days'] = (most_recent_date - df_up['last_review'])
df_up['days'] = df_up['days'].apply(lambda x: x.days) #extract days only
del df_up['last_review']



#######################################

#We are going to extract gender from the name of the owner

# check the last letter form the name
def gender_features(word):
    return {'last_letter': word[-1]}
gender_features('Shrek')



import random
from nltk.corpus import names
import nltk
nltk.download('names')


# Read the names from the files.
# Label each name with the corresponding gender.
male_names = [(name, 'male') for name in names.words('male.txt')]
female_names = [(name, 'female') for name in names.words('female.txt')]

# Combine the lists.
labeled_names = male_names + female_names

# Shuffle the list.
random.shuffle(labeled_names)

from nltk import NaiveBayesClassifier
# Extract the features using the `gender_features()` function.
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

# Split the dataset into train and test set
# Only for the purpose to teach the classifier to assign gender to the name
train_set, test_set = featuresets[50:], featuresets[:50]

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

neo_gender = classifier.classify(gender_features('Neo'))
trinity_gender = classifier.classify(gender_features('Trinity'))
print("Neo is most probably a {}.".format(neo_gender))
print("Trinity is most probably a {}.".format(trinity_gender))

# assign male or female
df_up['host_name'][0]
classifier.classify(gender_features(df_up['host_name'][0]))
df_up['gender'] = df_up['host_name'].apply(lambda x: classifier.classify(gender_features(x)))

# assign male = 1 or female = 0
def gender_to_numeric(x):
    if x=='female': return 0
    if x=='male':   return 1

# apply tp our outout
df_up['gender_num'] = df_up['gender'].apply(gender_to_numeric)

# deleting that we don't need
del df_up['host_name']
del df_up['gender']



#########################

#Let's work with categorical variables
df_up = pd.get_dummies(df_up)
###########################

#reset index
df_up.reset_index(drop=True, inplace=True)
###########################

#move price column to the first
df_up = df_up[['price'] + [x for x in df_up.columns if x != 'price']]



###########################
#BUILD A MODEL
###########################


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest #HERE
from sklearn.metrics import mean_absolute_error



# linear regression can make a negative output
# therefore we tranform our prices into ln

df_up['price'] = np.log(df_up['price'])


###### Build a baseline linear model
# So we can compare our results with the baseline model


# retrieve the array
data = df_up.values
# split into input and output elements
X, y = data[:, 1:], data[:, 0]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)

# check for min values
y_test.min()
yhat.min()

# transform back to the prices
y_test = np.exp(y_test)
yhat = np.exp(yhat)


# evaluate predictions based on MAE
mae_base = mean_absolute_error(y_test, yhat)

# evaluate predictions based on RMS
from sklearn.metrics import mean_squared_error
rms = mean_squared_error(y_test, yhat, squared=False)

print('MAE: %.3f' % mae_base)
print('RMSE: %.3f' % rms)

y_test[:10]
yhat[:10]



#########################
# evaluate model performance with outliers removed using isolation forest

# retrieve the array
data = df_up.values
# split into input and output elements
X, y = data[:, 1:], data[:, 0]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)



# identify outliers in the training dataset
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(X_train)

# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]

# summarize the shape of the updated training dataset (10%)
print(X_train.shape, y_train.shape)

# fit the model
model = LinearRegression()
model.fit(X_train, y_train)


# evaluate the model
yhat = model.predict(X_test)

y_test = np.exp(y_test)
yhat = np.exp(yhat)


# evaluate predictions
mae_iso_forest = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae_iso_forest)







#########################
# Minimum Covariance Determinant
# evaluate model performance with outliers removed using elliptical envelope

from sklearn.covariance import EllipticEnvelope


# retrieve the array
data = df_up.values
# split into input and output elements
X, y = data[:, 1:], data[:, 0]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)



ee = EllipticEnvelope(contamination=0.01)
yhat = ee.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)


y_test = np.exp(y_test)
yhat = np.exp(yhat)




# evaluate predictions
mae_cov = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae_cov)

print(y_test[:10])
print(yhat[:10])



#########################
# Local Outlier Factor

from sklearn.neighbors import LocalOutlierFactor


# retrieve the array
data = df_up.values
# split into input and output elements
X, y = data[:, 1:], data[:, 0]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)




# identify outliers in the training dataset
lof = LocalOutlierFactor()
yhat = lof.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)



y_test = np.exp(y_test)
yhat = np.exp(yhat)



# evaluate predictions
mae_lof = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae_lof)

rms = mean_squared_error(y_test, yhat, squared=False)
print('RMSE: %.3f' % rms)





#########################
# One-Class SVM
from sklearn.svm import OneClassSVM


# retrieve the array
data = df_up.values
# split into input and output elements
X, y = data[:, 1:], data[:, 0]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)


ee = OneClassSVM(nu=0.01)
yhat = ee.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)


y_test = np.exp(y_test)
yhat = np.exp(yhat)

# evaluate predictions
mae_oc_svm = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae_oc_svm)




######################################## Let's compare other models
# Decision Tree

from sklearn import tree


# retrieve the array
data = df_up.values
# split into input and output elements
X, y = data[:, 1:], data[:, 0]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# fit the model
model = tree.DecisionTreeRegressor(max_depth=5) # defined by trial and error
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)

y_test = np.exp(y_test)
yhat = np.exp(yhat)

# evaluate predictions
mae_dt = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae_dt)

rms = mean_squared_error(y_test, yhat, squared=False)

print('RMSE: %.3f' % rms)


#decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)

#SVM


from sklearn import svm

# retrieve the array
data = df_up.values
# split into input and output elements
X, y = data[:, 1:], data[:, 0]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# fit the model
model = svm.SVR(kernel="rbf") #kernel="Polynomial" #kernel="linear" # defined by trial and error
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)



y_test = np.exp(y_test)
yhat = np.exp(yhat)


# evaluate predictions
mae_svm = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae_svm)


rms = mean_squared_error(y_test, yhat, squared=False)
print('RMSE: %.3f' % rms)


######################################## 
##############OLS
import statsmodels.api as sm


#with outlires

data = df_up.values
# split into input and output elements
X, y = data[:, 1:], data[:, 0]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


X = X_train
y = y_train

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())






#without outlires

# retrieve the array
data = df_up.values
# split into input and output elements
X, y = data[:, 1:], data[:, 0]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)




# identify outliers in the training dataset
lof = LocalOutlierFactor()
yhat = lof.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)


X = X_train
y = y_train

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
























