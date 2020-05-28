"""
@author: Egesabanci
last update: 28 May 2020
"""

# Backpropagation for learning
# Updating weights to optimize loss value

def Backpropagation(inputs, weights, alpha, loss):
    for i in range(len(weights)):
        weights[i] = weights[i] + (alpha * loss * inputs[i])
    return weights