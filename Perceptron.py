"""
@author: Egesabanci
last update: 28 May 2020
"""

# Main file of the 'Perceptron' Module.
import numpy as np
from Loss import MSE, MAE, CrossEntropy, TAR_PRE
from BP import Backpropagation # Backpropagation function from BP.py

class Layer(object):
    # initialize
    def __init__(self, inputs, weights, target, 
                 bias = 1, ALPHA = 0.001, DECAY = 1e-5):
        self.inputs = inputs
        self.weights = weights
        self.target = target
        self.bias = bias
        self.ALPHA = ALPHA # learning rate
        self.DECAY = DECAY # decrease learning - each step
        
    # fit the model
    def fit(self, epochs, loss = 'MSE', verbose = True):
        self.epochs = epochs
        self.loss = loss
        self.verbose = verbose
    
        # in range -> n of epochs
        for iteration in range(self.epochs):
            # get prediction - multiplicate (weights, inputs)
            prediction = np.dot(self.weights, self.inputs) + self.bias
            
            # calculate loss - default = MSE
            if self.loss == 'MSE':
                loss_value = MSE(self.target, prediction, self.inputs)
            if self.loss == 'MAE':
                loss_value = MAE(self.target, prediction, self.inputs)
            if self.loss == 'CrossEntropy':
                loss_value = CrossEntropy(self.target, prediction, self.inputs)
            if self.loss == 'TAR_PRE':
                loss_value = TAR_PRE(self.target, prediction)
            
            # backpropagation - according to loss
            self.weights = Backpropagation(self.inputs, self.weights,
                                           self.ALPHA, loss_value)
            # decrease learning rate
            self.ALPHA -= self.DECAY
            
            # print loss, prediction, iteration if verbose = True
            if self.verbose == True:
                print(f'\nITERATION: {iteration + 1} \nPREDICTION: {round(prediction, 6)} \nLOSS: {abs(loss_value)}')
                if iteration == self.epochs - 1:
                    print(f'\nOPTIMIZED WEIGHTS AFTER {self.epochs} ITERATION: -> (LOSS: {loss_value})\n{self.weights}')
