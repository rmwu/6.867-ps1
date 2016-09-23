"""
6.867 Fall 2016 - Problem Set 1
Problem 1. Implement Gradient Descent
"""

import numpy as np
import loadFittingDataP1 as loadData
import loadParametersP1 as loadParams

##################################
# Main Functions
##################################

def gradient_descent(objective_function, gradient_function):
	"""
	Parameters
		objective_function
			Function used to calculate how good our guess
			is so far. In this case, our objective_function
			is a loss function to be minimized.
		gradient_function
			Function used to calculate the gradient at a given
			point.
	"""
	pass

def update(n, Theta, Y, x, eta):
	"""
	update(n, Theta, Y, x, eta) returns the new x
	value after xxxxxx

	Parameters
		n			xxxxxx
		Theta		xxxxxx
		Y			xxxxxx
		x			vector to be updated
		eta			constant step size
	"""
	pass

##################################
# Mathematical Implementations
##################################

def d_negative_gaussian(Sigma, x, mu):
	"""
	d_negative_gaussian(Sigma, x, mu) calculates the derivative
	of a negative gaussian function, which has the form:

		f(x) = [math]

	Parameters
		Sigma	covariance matrix
		x		x is x
		mu		mean
	"""
	pass

def d_quadratic_bowl(A, x, b):
	"""
	d_quadratic_bowl(A, x, b) calculates the derivative of a 
	quadratic bowl, which has the form:

		f(x) = 1/2 xT Ax - xT b

	Parameters
		A		positive definite matrix
		x
		b
	"""
	return A.dot(x) - b

def gradient_approximation(x, delta):
	"""
	gradient_approximation(x, delta, etc.) numerically approximates
	the gradient at a certain point.

	not sure what we need yet
	"""
	pass

##################################
# Input Handling
##################################

def get_inputs():
	"""
	get_inputs reads the input values from the data file.
	"""
	return loadData.getData()

def get_params():
	"""
	get_inputs reads the input values from the params file.
	"""
	return loadParams.getData()

def main():
	"""
	main function takes blah blah blah
	"""
	pass

