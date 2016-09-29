"""
6.867 Fall 2016 - Problem Set 1
Problem 4. Implement LASSO
"""
import sys
sys.path.append("..")

import numpy as np

import P2.lbf_regression as lbfr

def lasso_regularizer(weight_vector, reg_coef):
	"""
	Regularization term, sum-of-squares of weight vector
	"""
	return 0.5 * reg_coef * weight_vector.T.dot(weight_vector)

def lasso_weights(basis_functions, X, Y, lambd):
    """
    params:

    basis_functions: an iterable of functions, the first of which is the
    constant function 1

    X: the input data, as an iterable with individual x values as elements
    Y: the input data, as a rank-1 numpy tensor.
    lambd: lambda, the regularization parameter

    (see Bishop Equation 3.29)

    returns: max-likelihood weights for lasso
    """
    num_features = len(basis_functions)
    no_bias_term = basis_functions[1:]

    phi = lbfr.design_matrix(no_bias_term, X)  # design matrix
    avg_phi = phi.mean(0)
    z = phi - avg_phi  # phi, but centered

    yc = Y - Y.mean(0) # y centered

    weights_lasso = np.linalg.inv(lambd*np.identity(num_features-1) + z.T.dot(z)).dot(z.T).dot(yc)
    bias = Y.mean(0) - weights_lasso.dot(phi.mean(0))

    assert len(weights_lasso.shape) == 1
    assert weights_lasso.shape[0] == num_features-1
    return np.append(bias, weights_lasso)
