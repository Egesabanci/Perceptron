"""
@author: Egesabanci
last update: 28 May 2020
"""

import math
# Loss Functions
"""
-> Mean Squared Error
-> Mean Absolute Error
-> Cross Entropy
-> Target-Predict (highly recommended for value optimization)
"""

# Mean Squared Error
def MSE(target, predict, inputs):
    return ((target - predict) ** 2) / len(inputs)
    
# Mean Absolute Error
def MAE(target, predict, inputs):
    return (abs(target - predict)) / len(inputs)
    
# Cross Entropy
def CrossEntropy(target, predict, inputs):
    return (-1 / len(inputs)) * target * math.log(predict)
    
# Target - Predict
def TAR_PRE(target, predict):
    return target - predict