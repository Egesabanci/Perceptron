"""
@author: Egesabanci
last update: 28 May 2020
"""

# Test file for the 'Perceptron' module.
import random
import numpy as np

# --- Value optimization ---
inputs = np.array([random.randint(1, 10) for _ in range(10)])
weights = np.array([random.uniform(0, 1) for _ in range(len(inputs))])
target = np.array([25])
bias = 0.5
loss = 'TAR_PRE'
learning_rate = 1e-3
DECAY = 1e-6
epochs = 30

from Perceptron import Layer
model = Layer(inputs = inputs, weights = weights, target = target,
              bias = bias, ALPHA = learning_rate, DECAY = DECAY)
model.fit(epochs = epochs, loss = loss, verbose = True)
