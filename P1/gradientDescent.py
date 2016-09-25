"""
6.867 Fall 2016 - Problem Set 1
Problem 1. Implement Gradient Descent
"""
from __future__ import division

import numpy as np
import math

debug = True

##################################
# Main Functions
##################################

def generic_gradient_descent(x_init, objective, gradient, eta, threshold):
    """
    xxxx
    """
    iterations = 0

    current_x = x_init # this one is updated
    fx0 = 0 # last f(x)
    fx1 = float("inf") # current f(x)

    while True:
        current_x, grad = generic_update(gradient, current_x, eta)
        current_norm = np.linalg.norm(grad)

        fx1 = function(current_x)

        if debug:
            print("Gradient norm: {}\nCurrent X: {}\nObjective function: {}\n"\
            .format(current_norm, current_x, fx1))
            print("Past objective function: {}\n".format(fx0))

        if converge_delta_fx(fx0, fx1, threshold):
            break

        # update "past" objective function
        fx0 = fx1
        iterations += 1

    print("Converged after {} iterations\n".format(iterations))
    print("We updated to {}\n".format(current_x))
    return (current_x, fx1)

def gradient_descent(x_init, params, function, gradient, 
                     eta, threshold, grad_norm = False, delta = 0.05):
    """
    Parameters
        x_init
            2x1
        params
            2-tuple for parameters (for function). Either gauss_mean,
            gauss_cov OR A_quad_bowl, b_quad_bowl
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
        current_x, grad = update(gradient, params, current_x, eta)
        current_norm = np.linalg.norm(grad)

        # estimate gradient norm, gradient
        # est_slope, est_grad = central_difference(function, params, current_x, delta)
        est_slope, est_grad = 0, 0
        # calculate objective function
        fx1 = function(params, current_x)

        if debug:
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
    print("We updated to {}\n".format(current_x))
    return (current_x, fx1)

def generic_update(gradient, x, eta):
    grad = gradient(x)
    x_new = x - eta * grad

    return (x_new, grad)

def update(gradient, params, x, eta):
    """
    update(gradient, params, n, current_x, eta) returns the
    new x value after updating.

    Parameters
        gradient    gradient function
        param       parameters
        x           vector to be updated
        eta         constant step size
    """
    grad = gradient(params, x)
    x_new = x - eta * grad

    if debug:
        print("Gradient is {}\n".format(grad))

    return (x_new, grad)

##################################
# Mathematical Implementations
##################################

# gradient computing functions

def d_negative_gaussian(params, x):
    """
    d_negative_gaussian(params, x) calculates the derivative
    of a negative gaussian function, which has the form:

        f(x) = [math]

    Parameters
        params      contains (mu, sigma)
        n           number of samples
        x           current vector
    """
    mu, sigma = params
    n = sigma.shape[0]

    f_x = negative_gaussian(params, mu)

    return -f_x * np.linalg.inv(sigma).dot(x - mu)

def negative_gaussian(params, x):
    """
    negative_gaussian(params, x) calculates the negative gaussian
    function given inputs.

    Parameters
        params      contains (mu, Sigma)
        x           current vector
    """
    mu, Sigma = params
    n = Sigma.shape[0]

    gaussian_normalization = 1/(math.sqrt((2*math.pi)**n * np.linalg.det(Sigma)))
    
    exponent = -1/2 * (x - mu).T.dot(np.linalg.inv(Sigma)).dot(x - mu)

    return -1 * gaussian_normalization * math.exp(exponent)

def d_quadratic_bowl(params, x):
    """
    d_quadratic_bowl(params, x) calculates the derivative of a 
    quadratic bowl, which has the form:

        f(x) = 1/2 xT Ax - xT b

    Parameters
        params      contains (A, b)
        x           current vector
    """
    A, b = params
    return A.dot(x) - b

def quadratic_bowl(params, x):
    """
    quadratic_bowl(params, b) calculates the quadratic bowl function for given
    A, x, and b.

    Parameters
        params      contains (A, b)
        x           current vector
    """
    A, b = params
    return 0.5 * x.T.dot(A).dot(x) - x.T.dot(b)

def d_squared_error(params, theta):
    """
    d_squared_error(x, y, theta) calculates the gradient of J(x), which is
    the squared error:

        J(x) = || x theta - y|| ^2

    Parameters
        params      contains (x, y), n by n and n by 1
        theta       n by 1
    """
    x, y = params
    n, m = x.shape

    grad = 2 * x.T.dot(x.dot(theta) - y.reshape(n, 1))

    # print("Dot {} by {} - {} to get a {}".format(x.T.shape, x.dot(theta).shape, y.shape, grad.shape))
    return grad

def squared_error(params, theta):
    """
    squared_error(x, theta, y) returns the square error, defined
    as:

        J(theta) = || x theta - y || ^2

    Parameters
        x       n by n
        theta   n by 1
        y       n by 1
    """
    x, y = params

    return np.linalg.norm(x.dot(theta) - y) ** 2

# estimation and loss funtions

def central_difference(f, params, x, delta):
    """
    central_difference(f, params, x, delta) calculates an approximation of the gradient
    at a given point, for a given function f. The central difference is defined as:

        (f(x + delta/2) - f(x - delta/2)) / delta
    """
    # numpy should fix dimensions
    f_positiveDelta = f(params, x + delta/2)
    f_negativeDelta = f(params, x - delta/2)

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

##################################
# Stochastic GD
##################################
