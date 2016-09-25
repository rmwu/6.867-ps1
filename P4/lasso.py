"""
6.867 Fall 2016 - Problem Set 1
Problem 3. Implement Ridge Regression
"""

import numpy as np

def lasso_regularizer(weight_vector, reg_coef):
	"""
	Regularization term, sum-of-squares of weight vector
	"""
	return 0.5 * reg_coef * np.linalg.norm(weight_vector)