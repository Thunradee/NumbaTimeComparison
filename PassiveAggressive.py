# Author: Thunradee Tangsupakij
# Date: Nov 8th, 2020
# Description: This file implements Passive Aggressive algorithm. One class uses Numba library, the other one doesn't.
# They are to be used to compare running times.

import numpy as np
from numba import types
from numba.experimental import jitclass

# Binary Passive Aggressive class
class PassiveAggressive:

    def __init__(self, n_features, max_iter=1000):
        self.max_iter = max_iter
        self.weights = np.zeros(n_features, dtype=np.float64)

    def fit(self, X, y):
        '''
        Train classifier
        :param X: A d-dimensional array
            Features
        :param y: A one-dimensional array
            Labels
        '''
        for epoch in range(self.max_iter):
            # For each data point in the training set
            for i, x in enumerate(X):
                # Predict using the current weights
                a = np.dot(self.weights, x)
                pred = int(np.sign(a))
                # If the prediction is wrong
                if pred != y[i]:
                    # calculate L2 norm square
                    l2sq = np.linalg.norm(x, ord=2)**2
                    # calculate learning rate
                    learning_rate = (1 - (y[i] * a)) / l2sq
                    # Update the weights
                    self.weights += (learning_rate * y[i]) * x

    def predict(self, X):
        '''
        Predict label of the given features
        :param X:  A d-dimensional array
            Features
        :return: A one-dimensional array
            Predictions
        '''
        # Initialize prediction list
        predictions = np.zeros(len(X), dtype=np.int32)
        # For each data point in the testing set
        for i, x in enumerate(X):
            # Predict a label
            pred = np.sign(np.dot(self.weights, x))
            # Add to the prediction list
            predictions[i] = int(pred)
        # Return the prediction list
        return predictions


# Binary Passive Aggressive class using Numba
@jitclass([('max_iter', types.int64), ('weights', types.float64[:])])
class NumbaPassiveAggressive:

    def __init__(self, n_features, max_iter=1000):
        self.max_iter = max_iter
        self.weights = np.zeros(n_features, dtype=np.float64)

    def fit(self, X, y):
        '''
        Train classifier
        :param X: A d-dimensional array
            Features
        :param y: A one-dimensional array
            Labels
        '''
        for epoch in range(self.max_iter):
            # For each data point in the training set
            for i, x in enumerate(X):
                # Predict using the current weights
                a = np.dot(self.weights, x)
                pred = int(np.sign(a))
                # If the prediction is wrong
                if pred != y[i]:
                    # calculate L2 norm square
                    l2sq = np.linalg.norm(x, ord=2)**2
                    # calculate learning rate
                    learning_rate = (1 - (y[i] * a)) / l2sq
                    # Update the weights
                    self.weights += (learning_rate * y[i]) * x

    def predict(self, X):
        '''
        Predict label of the given features
        :param X:  A d-dimensional array
            Features
        :return: A one-dimensional array
            Predictions
        '''
        # Initialize prediction list
        predictions = np.zeros(len(X), dtype=np.int32)
        # For each data point in the testing set
        for i, x in enumerate(X):
            # Predict a label
            pred = np.sign(np.dot(self.weights, x))
            # Add to the prediction list
            predictions[i] = int(pred)
        # Return the prediction list
        return predictions