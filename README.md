# Perceptron Module

**Main Structure**

Single neuron Perceptron with
backpropagation process.
Can be used for "value optimization".
Loss value is calculating according
to 'target'.

---

Layer (from Perceptron.py):

	PARAMETERS:
	
		-> inputs
		
		-> weights
		
		-> target (target value for adjust weights)
		
		-> bias
		
		-> ALPHA (learning rate)
		
		-> DECAY (decreasing the learning rate)
		

fit (from Layer.fit()):

	PARAMETERS:
	
		-> epochs
		
		-> loss (loss function):
		
			--> 'MSE', 'MAE', 'CrossEntropy', 'TAR_PRE'
			
		-> verbose (for information of the process) -> 'True' or 'False'
		

Backpropagation (from BP.py):

	-> Equation = New Weight = Old Weight + ALPHA * (loss value) * Current input
	
	-> After feed forward; backpropagate and update the weights
	   with backpropagation equation.(For each weight)
	   
	-> ALPHA = learning rate


Loss (from Loss.py):

	* Mean Squared Error = sum of((target - predict) ** 2) / number of inputs 
	
	* Mean Absolute Error = sum of(abs(target - predict)) / number of inputs
	
	* Cross Entropy = (-1 / len(inputs)) * sum of(target.log(predict))
	
	* TAR_PRE (target - predict) Error = (target - predict) -> (highly recommended) 
	
-----

--- FILES ---

-> Perceptron.py (Main file - include Layer class)

-> BP.py (Backpropagation file - include backpropagation function)

-> Loss.py (Loss functions - include loss functions)


---
**@author: Egesabanci**

**last update: 28 May 2020**

---
