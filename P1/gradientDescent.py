"""
6.867 Fall 2016 - Problem Set 1
Problem 1. Implement Gradient Descent
"""
from __future__ import division

import numpy as np
import math

##################################
# Main Functions
##################################

def gradient_descent(x_init, params, loss, function, gradient, 
                     eta, threshold, grad_norm = False, delta = 0.05):
    """
    Parameters
        x_init
            2x1
        params
            2-tuple for parameters (for function). Either gauss_mean,
            gauss_cov OR A_quad_bowl, b_quad_bowl
        loss
            function used to calculate how good our guess
            is so far. In this case, our objective_function
            is a loss function to be minimized.
        gradient
            function used to calculate the gradient at a given
            point.
        eta
            constant step size
        threshold
            convergence threshold
        grad_norm
            True if we compute gradient norm, False if
            we compute change in the vector
    """
    n, m = x_init.shape

    current_x = x_init
    fx0 = 0 # last f(x)
    fx1 = float("inf") # current f(x)

    iterations = 0 # count iterations until converge

    while True:
        # update step
        current_x, grad = update(gradient, params, n, current_x, eta)
        current_norm = np.linalg.norm(grad)

        # estimate gradient norm, gradient
        est_slope, est_grad = central_difference(function, params, n, current_x, delta)
        # calculate objective function
        fx1 = function(params, n, current_x)

        print("Gradient norm: {}\nCurrent X: {}\nObjective function: {}\nEstimated gradient: {}"\
            .format(current_norm, current_x, fx1, est_grad))
        print("Past objective function: {}\n".format(fx0))

        # check for convergence
        if grad_norm and converge_grad_norm(grad, threshold):
            break

        elif not grad_norm and converge_delta_fx(fx0, fx1, threshold):
            break
        
        # update "past" objective function
        fx0 = fx1
        iterations += 1

    print("Converged after {} iterations\n".format(iterations))
    return (current_x, fx1)


def update(gradient, params, n, x, eta):
    """
    update(gradient, params, n, current_x, eta) returns the
    new x value after updating.

    Parameters
        gradient    gradient function
        param       parameters
        n           number of samples
        x           vector to be updated
        eta         constant step size
    """
    grad = gradient(params, n, x)
    x_new = x - eta * grad

    print("\nupdating x from\n{}\nto\n{}\n".format(x, x_new))
    print("gradient=\n{}\n".format(grad))
    # print('k')

    return (x_new, grad)

##################################
# Mathematical Implementations
##################################

# gradient computing functions

def d_negative_gaussian(params, n, x):
    """
    d_negative_gaussian(params, n, x) calculates the derivative
    of a negative gaussian function, which has the form:

        f(x) = [math]

    Parameters
        params      contains (mu, sigma)
        n           number of samples
        x           current vector
    """
    mu, sigma = params
    f_x = negative_gaussian(params, n, mu)

    return -f_x * np.linalg.inv(sigma).dot(x - mu)

def negative_gaussian(params, n, x):
    """
    negative_gaussian(params, n, x) calculates the negative gaussian
    function given inputs.

    Parameters
        params      contains (mu, Sigma)
        n           number of samples
        x           current vector
    """
    mu, Sigma = params

    gaussian_normalization = 1/(math.sqrt((2*math.pi)**n * np.linalg.det(Sigma)))
    
    exponent = -1/2 * (x - mu).T.dot(np.linalg.inv(Sigma)).dot(x - mu)

    return -1 * gaussian_normalization * math.exp(exponent)

def d_quadratic_bowl(params, n, x):
    """
    d_quadratic_bowl(params, n, x) calculates the derivative of a 
    quadratic bowl, which has the form:

        f(x) = 1/2 xT Ax - xT b

    Parameters
        params      contains (A, b)
        n           number of samples
        x           current vector
    """
    A, b = params
    return A.dot(x) - b

def quadratic_bowl(params, n, x):
    """
    quadratic_bowl(A, x, b) calculates the quadratic bowl function for given
    A, x, and b.

    Parameters
        params      contains (A, b)
        n           number of samples
        x           current vector
    """
    A, b = params
    return 0.5 * x.T.dot(A).dot(x) - x.T.dot(b)

# estimation and loss funtions

def least_square_error(X, Theta, y):
    """
    least_square_error(X, Theta, y) returns the least square error, defined
    as:

        J(Theta) = || X Theta - y || ^2
    """
    pass

def central_difference(f, params, n, x, delta):
    """
    central_difference(f, params, n, x, delta) calculates an approximation of the gradient
    at a given point, for a given function f. The central difference is defined as:

        (f(x + delta/2) - f(x - delta/2)) / delta
    """
    # numpy should fix dimensions
    f_positiveDelta = f(params, n, x + delta/2)
    f_negativeDelta = f(params, n, x - delta/2)

    est_slope = (f_positiveDelta - f_negativeDelta) / delta

    est_grad = np.array([[delta],[f_positiveDelta -  f_negativeDelta]])
    est_grad = (est_grad / np.linalg.norm(est_grad)) * est_slope

    return (est_slope, est_grad)

def converge_grad_norm(grad, threshold):
    """
    converge_grad_norm(grad, threshold) returns True if the gradient's norm
    is under threshold, False otherwise.

    Parameters
        grad        must be a vector
        threshold   must be a scalar
    """
    return np.linalg.norm(grad) < threshold

def converge_delta_fx(fx0, fx1, threshold):
    """
    converge_delta_fx(fx0, fx1, threshold) returns True if the change in f(x)
    between two success updates is less than threshold, False otherwise.

    Parameters
        fx0         must be a scalar
        fx1         must be a scalar
        threshold   must be a scalar
    """
    return abs(fx1 - fx0) < threshold
