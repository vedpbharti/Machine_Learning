#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:42:52 2018

@author: ved
"""
# Import required packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve


# Import data
colname = ["F" + str(i) for i in range(0,296)] #set column name to F_xx
df = pd.read_csv("sample.csv", header = None, names = colname )

# Target class distribution
df.F295.value_counts().plot(kind = "bar")
# check and remove features with only value
append_count = []
for i in df:
	count = df[i].value_counts()
	append_count.append(count)
# 59,179,268,269,270,271,272,273,274,275,276 are the columns with only 0 value
df = df.loc[:, (df != df.iloc[0]).any()]

# check the number of zeros and if there is null in dataframe
np.count_nonzero(df)
df.isnull().sum().sum()

# check correlation matrix
corr = df.corr()
upper_portion = corr.where(np.triu(np.ones(corr.shape), k = 1).astype(np.bool))
to_drop = [column for column in upper_portion.columns if any(upper_portion[column].abs() > 0.30)]
# Dataframe to hold correlated pairs
record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])
for column in to_drop:
	corr_features = list(upper_portion.index[upper_portion[column].abs() > 0.30])
	corr_values = list(upper_portion[column][upper_portion[column].abs() > 0.30])
	drop_features = [column for _ in range(len(corr_features))]  
	temp = pd.DataFrame.from_dict({'drop_feature': drop_features,
                               	'corr_feature': corr_features,
                               	'corr_value': corr_values})
record_collinear = record_collinear.append(temp, ignore_index = True)


# Encode categorical data (Target class)
LabelEncoder = LabelEncoder()
df.F295 = LabelEncoder.fit_transform(df.F295)
# after code the classes 
#C(2)      46882
#D(3)      9279
#B(1)      6602
#E(4)      2507
#A(0)      867

# Remove column with low variance
def VarianceThreshold_selector(data):
	columns = data.columns
	#remove features with 0 and 1 in more than 30% in column
	selector = VarianceThreshold(.3)
	selector.fit_transform(data)
	labels = [columns[x] for x in selector.get_support(indices=True) if x]
	return pd.DataFrame(selector.fit_transform(data), columns=labels)
df_mod = VarianceThreshold_selector(df)

# Check target class plot
df_mod.F295.value_counts().plot(kind='bar')
# Correlation matrix
corr = df_mod.corr()
sns.heatmap(corr)
sns.pairplot(df_mod)

# Setting up target and dependant variables
X = df_mod.iloc[:,:-1].values
y = df_mod.iloc[:, 6].values

# Splitting dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature scaling
SC = StandardScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)

# Handling Imbalanced target class and checking the impact on RandomForest and Logistic regression classifier
# http://contrib.scikit-learn.org/imbalanced-learn/stable/introduction.html 
algorithms = (SMOTE(),ADASYN(), RandomUnderSampler(random_state=0), NearMiss(random_state=0,version=1)) #ClusterCentroids(random_state=0),
algr = ['SMOTE', 'ADASYN', 'RandomUnderSampler', 'NearMiss'] # 'ClusterCentroids',
i=0 
for algo in algorithms:
    print('--------------------------')
    print('### Count of class values using - ' + algr[i])
    X_resampled, y_resampled = algo.fit_sample(X_train, y_train)
    print(sorted(Counter(y_resampled).items()))
    classifier = ( DecisionTreeClassifier(), RandomForestClassifier(n_estimators= 50, criterion= 'entropy', random_state= 0), LogisticRegression(random_state = 0))
    mod = ['DecisionTreeClassifier','RandomForestClassifier', 'LogisticRegression']
    j = 0
    for model in classifier:
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print('--------------------------')
        print('### Model evaluation using classifier - ' + mod[j])
        print('                          ')
        print(classification_report(y_test, y_pred))
        accuracies = cross_val_score(estimator = model, X = X_resampled, y = y_resampled, cv =5)
        print('### 5 Fold cross validation results - ' + mod[j])
        print (" mean accuracy :: {} ".format(accuracies.mean()))
        print (" mean std :: {} ".format(accuracies.std()))
        j=j+1
    print('--------------------------')
    i=i+1

# As randomforest gives better results, using grid find out best parameters
# Grid search Randomforest
X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train)
classifier = RandomForestClassifier(n_estimators= 200, criterion= 'entropy', random_state= 0)

parameters = [{'n_estimators':[30,50,100,150], 'max_features': ['auto', 'sqrt', 'log2'], 'criterion': ['entropy','gini']}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 5, n_jobs = -1)
grid_search = grid_search.fit(X_resampled, y_resampled)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
# hence best estimator number for our problem is 150, 
classifier_rf = RandomForestClassifier(n_estimators= 200, criterion= 'gini', random_state= 0)
classifier_rf.fit(X_resampled, y_resampled)
y_pred = classifier_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

# Learning curve 
train_sizes, train_scores, valid_scores = learning_curve(RandomForestClassifier(n_estimators= 200, criterion= 'gini', random_state= 0), X_resampled, y_resampled, scoring = 'accuracy', n_jobs = -1, train_sizes=np.linspace(0.01, 1.0, 20))
# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
# Create means and standard deviations of test set scores
test_mean = np.mean(valid_scores, axis=1)
test_std = np.std(valid_scores, axis=1)
# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()

# However, confusion matrix suggest even though the accuracy is almost 88% but the classifier is not doing good job predicting all the class
# One class prediction is completely dominating the other 3 classes. 

# UnderSampling and way to avoid accuracy paradox by not only checking the accuracy but the class level correct prediction performance. 
X_resampled, y_resampled = ClusterCentroids(random_state=0).fit_sample(X_train, y_train)

# Grid search for SVC tuning 
classifier = SVC( kernel = 'rbf', random_state = 0)
parameters = [{'C':[1,10,100,1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 5, n_jobs = -1)
grid_search = grid_search.fit(X_resampled, y_resampled)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# SVC with tuned parameters including cost function C (Penalty parameter of the error term to prevent overfitting). 
classifier_svc = SVC( kernel = 'rbf', random_state = 0, gamma = 0.1, C = 100)
classifier_svc.fit(X_resampled, y_resampled)
y_pred = classifier_svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))


# Grid search for KNNclassifier tuning 
classifier = KNeighborsClassifier()
parameters = [{'metric':['minkowski','euclidean','manhattan'], 'weights': ['uniform','distance'], 'n_neighbors':  np.arange(5,10)}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 5, n_jobs = -1)
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 5, n_jobs = -1)
grid_search = grid_search.fit(X_resampled, y_resampled)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

# KNN classifier with tuned parameters
classifier_knn = KNeighborsClassifier(metric = 'minkowski', weights = 'uniform', n_neighbors = 9)
classifier_knn.fit(X_resampled, y_resampled)
y_pred = classifier_knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

# Learning curve 
train_sizes, train_scores, valid_scores = learning_curve( KNeighborsClassifier(metric = 'minkowski', weights = 'uniform', n_neighbors = 9), X_resampled, y_resampled, scoring = 'accuracy', n_jobs = -1, train_sizes=np.linspace(0.01, 1.0, 10))
# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
# Create means and standard deviations of test set scores
test_mean = np.mean(valid_scores, axis=1)
test_std = np.std(valid_scores, axis=1)
# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


# Merge two classifier Randomforest and KNN
from brew.base import Ensemble
from brew.base import EnsembleClassifier
from brew.combination.combiner import Combiner
# Random Sampling  
X_resampled, y_resampled = RandomUnderSampler(random_state=0).fit_sample(X_train, y_train)
clfs = [classifier_rf, classifier_knn]
ens = Ensemble(classifiers = clfs)
comb = Combiner(rule='max')
eclf = EnsembleClassifier(ensemble=ens, combiner=Combiner('mean'))
eclf.fit(X_resampled, y_resampled)
y_pred = eclf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))


# PCA Using feature reduction technique
# Check how many components needed in a way which will express maximum variance
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train_pca = pca.fit(X_train)
X_test_pca = pca.fit(X_test)
explained_variance = pca.explained_variance_ratio_

# Transforming to two features
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)
classifier = RandomForestClassifier(criterion='gini',n_estimators= 150 )
classifier.fit(X_resampled, y_resampled)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))



# Neural Network
from keras.models import Sequential #Sequential Models
from keras.layers import Dense #Dense Fully Connected Layer Type
# Perform one-hot-encoding on the target data
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
Y = encoder.fit_transform(y_resampled) # encode the target label
y_test_cat = encoder.fit_transform(y_test) # encode the target test label
X = SC.fit_transform(X_resampled) #scaling the input features.

# Perform one-hot-encoding on the target data
model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(5, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # categorical crossentropy becasue there are more than binary classes


model.fit(X, Y, batch_size = 10, nb_epoch = 25)
scores = model.evaluate(X_test, y_test_cat)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
