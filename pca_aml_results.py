# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:42:18 2019

@author: monicai
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:13:56 2019

@author: monicai
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

np.random.seed(seed=42)
#new data
new_data = pd.read_csv("processed_aml_data_2.csv")
X = new_data.iloc[:, 1:1011]
y = new_data.iloc[:, 1011]

#convert X and y to numpy
X = X.values
y = y.values

#split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(X, y, train_size = 0.8, test_size = 0.20, random_state = 42)


# Average CV score on the training set was:0.8138888888888889
exported_pipeline = make_pipeline(
    PCA(iterated_power="auto", n_components=4, random_state=42, svd_solver="auto"),
    RandomForestClassifier(bootstrap=True, criterion="gini", max_features="auto", min_samples_leaf=1, min_samples_split=2, n_estimators=100, random_state=42))

exported_pipeline.fit(x_train, y_train)
test_results = exported_pipeline.predict(x_val)
test_probs = exported_pipeline.predict_proba(x_val)[:, 1]

train_results = exported_pipeline.predict(x_train)
train_probs = exported_pipeline.predict_proba(x_train)[:, 1] 

from sklearn import metrics

#test
test_acc = metrics.accuracy_score(y_val, test_results)
test_balanced_acc = metrics.balanced_accuracy_score(y_val, test_results)
fpr, tpr, thresholds = metrics.roc_curve(y_val, test_probs)
test_auc = metrics.roc_auc_score(y_val, test_probs)
precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_val, test_results)
conf_mat = metrics.confusion_matrix(y_val, test_results)
tn, fp, fn, tp = conf_mat.ravel()
precision = (tp * 1.0)/(tp + fp)  #percent of all that were idenfied as positive that were actually positive
recall_sens = (tp * 1.0)/(tp + fn) #percent of all positives that were correctly identified as positives
specificity = (tn * 1.0)/(tn + fp)  #percent of all negatives that were correctly identified as negatives
lr_pos = recall_sens/(1 - specificity)
lr_neg = (1-recall_sens)/specificity


#train
train_cm = metrics.confusion_matrix(y_train, train_results)
train_acc = metrics.accuracy_score(y_train, train_results)
train_auc = metrics.roc_auc_score(y_train, train_probs)