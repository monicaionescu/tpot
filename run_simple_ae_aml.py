# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 21:31:13 2019

@author: monicai
"""

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split, cross_val_score
# from tpot.config.classifier_nn import classifier_config_nn

from sklearn.pipeline import make_pipeline
# from tpot.config import classifier_config_dict_light
from tpot.config import classifier_config_dict
from sklearn.neighbors import KNeighborsClassifier


import pandas as pd
import numpy as np
import os
import glob

import time

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from tpot.builtins import SimpleAutoencoder
pd.options.mode.chained_assignment = None

# personal_config = classifier_config_dict_light
personal_config = classifier_config_dict
personal_config['tpot.builtins.SimpleAutoencoder'] = {
    'encoding_dim': [10, 50, 100],
    'activation': ['relu'],
    'optimizer': ['adadelta', "SGD", "Adam", "Adamax", "Nadam"],
    'loss':['binary_crossentropy', 'hinge', 'mean_squared_error', 'mean_absolute_error'],
    'epochs':[50, 100],
    'batch_size':[10, 15]
}

#new data
new_data = pd.read_csv("processed_aml_data_2.csv")
X = new_data.iloc[:, 1:1011]
y = new_data.iloc[:, 1011]

#convert X and y to numpy
X = X.values
y = y.values

#split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(X, y, train_size = 0.8, test_size = 0.20, random_state = 42)

gens = 50
pop_size = 100
template = "SimpleAutoencoder-RandomForestClassifier"

tpot = TPOTClassifier(generations=gens, config_dict=personal_config,
                        population_size=pop_size, verbosity=3,
                        template = template, n_jobs = 8)

start = time.time()
tpot.fit(x_train, y_train)
end = time.time()
run_time = end - start
print(run_time)

#export results
final_pipeline = tpot.fitted_pipeline_
training_acc = tpot.score(x_train, y_train)
testing_acc = tpot.score(x_val, y_val)

results_dict = {'generations': [gens], 'population_size': [pop_size],
                'training_acc': [training_acc], 'testing_acc': [testing_acc], 'training_time': [run_time], "pipeline": [final_pipeline]}
simple_aml_results_df = pd.DataFrame(data=results_dict)

simple_aml_results_df.to_csv('simple_aml_results.csv',index=False)









