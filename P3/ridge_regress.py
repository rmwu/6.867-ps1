"""
6.867 Fall 2016 - Problem Set 1
Problem 3. Implement Ridge Regression
"""
import numpy as np
import sys
sys.path.append("..")

import P2.poly_basis_grad_descent as pbgd

def weight_regularizer(weight_vector, reg_coef):
	"""
	Regularization term, sum-of-squares of weight vector
	"""
	return 0.5 * reg_coef * weight_vector.T.dot(weight_vector)

def ridge_error(weight_vector, X, Y):
	return poly_basis_square_sum_error + weight_regularizer(weight_vector)

def ridge_error_ml(phi, x, reg_coef = 0.25):
	"""
	max likelihood weight

	x data vector
	"""
	n = phi.shape[0]
	return np.linalg.inv(reg_coef * np.identity(n)+ 
	    phi.T.dot(phi)).dot(phi.T.dot(x))