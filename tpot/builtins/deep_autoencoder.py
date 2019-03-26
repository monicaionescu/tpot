# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:31:12 2019

@author: monicai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from numpy import array
from numpy import argmax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import check_array, check_X_y
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None

class SimpleAutoencoder(BaseEstimator, TransformerMixin):
    def __init__(self, num_layers, encoding_dims, activation, optimizer, loss, epochs, batch_size, random_state=42):
        self.num_layers = num_layers
        self.encoding_dims = encoding_dims  #list of integers
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state



    def fit(self, X, y):
        """Do nothing and return the estimator unchanged
        This method is just there to implement the usual API and hence
        work in pipelines.
        Parameters
        ----------
        X : array-like
        """
        X, y = check_X_y(X, y, accept_sparse=True, dtype=None)
        x_train, x_val, _, _ = train_test_split(
            X, y, test_size=0.25, train_size=0.75, random_state=self.random_state)


        X_width = x_train.shape[1]
        
        self.input_placeholder = Input(shape = (X_width, ))    #input placeholder
        
        if self.num_layers == 2:
            self.encoded1 = Dense(self.encoding_dims[0], activation = self.activation)(self.input_placeholder)
            self.encoded2 = Dense(self.encoding_dims[1], activation = self.activation)(self.encoded1)
            
            self.decoded1 = Dense(self.encoding_dims[0], activation = self.activation)(self.encoded2)
            self.decoded2 = Dense(self.X_width, activation = 'sigmoid')(self.decoded1)
                      

            #define autoencoder model object
            self.autoencoder = Model(self.input_placeholder, self.decoded2)
            self.autoencoder.compile(optimizer = self.optimizer, loss = self.loss)
            self.autoencoder.fit(x_train, x_train,
                             epochs = self.epochs,
                             verbose=0,
                             batch_size = self.batch_size,
                             shuffle = True,
                             validation_data = (x_val, x_val))
            #define separate encoder model object
            self.encoder = Model(self.input_placeholder, self.encoded)
            
        elif self.num_layers == 3:
            self.encoded1 = Dense(self.encoding_dims[0], activation = self.activation)(self.input_placeholder)   
            self.encoded2 = Dense(self.encoding_dims[1], activation = self.activation)(self.encoded1)
            self.encoded3 = Dense(self.encoding_dims[2], activation = self.activation)(self.encoded2)
            
            self.decoded1 = Dense(self.encoding_dims[1], activation = self.activation)(self.encoded3)
            self.decoded2 = Dense(self.encoding_dims[0], activation = self.activation)(self.decoded1)
            self.decoded3 = Dense(self.X_width, activation = 'sigmoid')(self.decoded2)
            
            #define autoencoder model object
            self.autoencoder = Model(self.input_placeholder, self.decoded3)
            
            self.autoencoder.compile(optimizer = self.optimizer, loss = self.loss)
        
            self.autoencoder.fit(x_train, x_train, 
                             epochs = self.epochs, 
                             batch_size = self.batch_size, 
                             shuffle = True, 
                             validation_data = (x_val, x_val))
        
            #define separate encoder model object
            self.encoder = Model(self.input_placeholder, self.encoded3)
            
        return self


    def transform(self, X):
        #compile autoencoder model
        X = check_array(X, accept_sparse='csr')

        encoded_preds = self.encoder.predict(X)
        return encoded_preds
