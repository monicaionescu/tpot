# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:04:21 2019

@author: monicai
"""

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split, cross_val_score
# from tpot.config.classifier_nn import classifier_config_nn

from sklearn.pipeline import make_pipeline
# from tpot.config import classifier_config_dict_light
from tpot.config import classifier_config_dict
from sklearn.neighbors import KNeighborsClassifier

from keras import regularizers


import pandas as pd
import numpy as np
import os
import glob

import time

from tpot.builtins import SparseAutoencoder

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
pd.options.mode.chained_assignment = None

# personal_config = classifier_config_dict_light
personal_config = classifier_config_dict
personal_config['tpot.builtins.SparseAutoencoder'] = {
    'regularizer':['l1', 'l2'],                              #['l1', 'l2', 'l1_l2'],
    'reg_constant':[10e-4, 10e-5],                            #[10e-4, 10e-5, 10e-6],
    'encoding_dim': [10, 30, 50, 100],                            #[10, 50, 100, 500, 1000, 1100],
    'activation': ['relu'],
    'optimizer': ['adadelta'],                         #['adadelta', "SGD", "Adam", "Adamax", "Nadam"],
    'loss': ['mean_absolute_error'],                   #['binary_crossentropy', 'hinge', 'mean_squared_error', 'mean_absolute_error'],
    'epochs':[500],
    'batch_size':[15]
}

# =============================================================================
personal_config['sklearn.ensemble.RandomForestClassifier'] = {
         'n_estimators': [100],
         'criterion': ["gini"],
         'max_features': ['auto'],
         'min_samples_split': [2],
         'min_samples_leaf':  [1],
         'random_state': [42],
         'bootstrap': [True]}
# =============================================================================

# =============================================================================
# personal_config['sklearn.linear_model.LogisticRegression'] = {
#         'penalty': ['l2'], #["l1", "l2"],
#         'C': [1e-4],   #[1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
#         'dual': [False],
#         'random_state': [42]}
# =============================================================================

np.random.seed(seed=42)


#new data
new_data = pd.read_csv("processed_aml_data_2.csv")
X = new_data.iloc[:, 1:1011]
y = new_data.iloc[:, 1011]


# =============================================================================
# #new data
# new_data = pd.read_csv("processed_aml_data_3.csv")
# X = new_data.iloc[:, 0:12625]
# y = new_data.iloc[:, 12625]
# 
# =============================================================================

#convert X and y to numpy
X = X.values
y = y.values


#split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(X, y, train_size = 0.8, test_size = 0.20, random_state = 42)

gens = 5
pop_size = 10
template = 'SparseAutoencoder-RandomForestClassifier'
#template = 'SparseAutoencoder-LogisticRegression'
#template = 'SparseAutoencoder-XGBClassifier'

tpot = TPOTClassifier(generations=gens, config_dict=personal_config,
                        population_size=pop_size, verbosity=3, random_state = 42,
                        template = template)

start = time.time()
tpot.fit(x_train, y_train)
end = time.time()
run_time = end - start

final_pipeline = tpot.fitted_pipeline_

training_acc = tpot.score(x_train, y_train)

testing_acc = tpot.score(x_val, y_val)

results_dict = results_dict = {'generations': [gens], 'population_size': [pop_size],
                'training_acc': [training_acc], 'testing_acc': [testing_acc], 
                'training_time': [run_time], "pipeline": [final_pipeline]}

sparse_aml_results_df = pd.DataFrame(data=results_dict)

sparse_aml_results_df.to_csv('sparse_aml_results.csv',index=False)

tpot.export('sparse_ae_pipeline.py')









