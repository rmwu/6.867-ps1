"""
6.867 Fall 2016 - Problem Set 1
Problem 1. Implement Gradient Descent
"""
from __future__ import division
import numpy as np
from math_functions import *

import pylab as pl
import matplotlib.pyplot as plt

debug = True

##################################
# Input handling
##################################

def get_raw_data():
    
    # load the fitting data for X and y and return as elements of a tuple
    # X is a 100 by 10 matrix and y is a vector of length 100
    # Each corresponding row for X and y represents a single data sample

    X = pl.loadtxt('fittingdatap1_x.txt')
    y = pl.loadtxt('fittingdatap1_y.txt')

    return (X,y)

def get_raw_params():

    # load the parameters for the negative Gaussian function and quadratic bowl function
    # return a tuple that contains parameters for Gaussian mean, Gaussian covariance,
    # A and b for quadratic bowl in order

    data = pl.loadtxt('parametersp1.txt')

    gaussMean = data[0,:]
    gaussCov = data[1:3,:]

    quadBowlA = data[3:5,:]
    quadBowlb = data[5,:]

    return (gaussMean,gaussCov,quadBowlA,quadBowlb) 

def get_gaussian_params():
    mu, sigma = get_raw_params()[:2]
    mu = np.array(mu).reshape(2, 1)
    sigma = np.array(sigma)

    return (mu, sigma)

def get_quad_params():
    a, b = get_raw_params()[2:4]
    b = np.array(b).reshape(2, 1)
    a = np.array(a)

    return (a, b)

def get_data():
    x, y = get_raw_data()
    x, y = np.array(x).reshape(100, 10), np.array(y).reshape(100, 1)
    return (x, y)

mu, sigma = get_gaussian_params()
a, b = get_quad_params()
x_data, y_data = get_data()

##################################
# Mathematical Implementations
##################################

def d_negative_gaussian(x):
    """
    d_negative_gaussian(x) calculates the derivative
    of a negative gaussian function.

    Parameters
        x           current vector
    """
    n, _ = sigma.shape
    f_x = negative_gaussian(mu)

    return -f_x * np.linalg.inv(sigma).dot(x - mu)

def negative_gaussian(x):
    """
    negative_gaussian(x) calculates the negative gaussian
    function given inputs.

    Parameters
        x           current vector
    """
    n = sigma.shape[0]

    gaussian_normalization = 1/(math.sqrt((2*math.pi)**n * np.linalg.det(sigma)))
    
    exponent = -1/2 * (x - mu).T.dot(np.linalg.inv(sigma)).dot(x - mu)

    return -1 * gaussian_normalization * math.exp(exponent)

def d_quadratic_bowl(x):
    """
    d_quadratic_bowl(x) calculates the derivative of a 
    quadratic bowl, which has the form:

        del f(x) = Ax - b

    Parameters
        x           n by 1
    """
    return a.dot(x) - b

def quadratic_bowl(x):
    """
    quadratic_bowl(x) calculates the quadratic bowl function,
    which has the form:

        f(x) = 1/2 xT Ax - xT b

    Parameters
        x           n by 1
    """
    return 0.5 * x.T.dot(a).dot(x) - x.T.dot(b)

def d_squared_error(theta):
    """
    d_squared_error(theta) calculates the gradient of J(x), which is
    the squared error:

        J(x) = || x theta - y|| ^2

    Parameters
        theta       n by 1
    """
    return 2 * x_data.T.dot(x_data.dot(theta) - y_data)

def squared_error(theta):
    """
    squared_error(theta) returns the square error, defined
    as:

        J(theta) = || x theta - y || ^2

    Parameters
        x       n by n
        theta   n by 1
        y       n by 1
    """
    n, m = x_data.shape
    return np.linalg.norm(x_data.dot(theta) - y_data) ** 2

def d_stochastic_error(theta, i):
    xi = x_data[i].reshape(10, 1)
    yi = y_data[i]

    return 2 * xi * (xi.T.dot(theta.reshape(10, 1))-yi)

def stochastic_error(theta, i):
    """
    calculates

        J(x_t) = || x(i).T theta_t - y(i) || ^2

    where t represents the iteration, and i represents the sample index
    """
    xiT = x_data[i].reshape(10, 1).T
    yi = y_data[i]

    # print("x {} and y{}".format(xiT, yi))
    # print("stoch error{}".format((xiT.dot(theta.reshape(10, 1))-yi) ** 2))
    return (xiT.dot(theta.reshape(10, 1))-yi) ** 2 


##################################
# Main Functions
##################################

def gradient_descent(x_init, objective, gradient, eta, threshold, delta, conv_by_grad,
                     stochastic = False):
    """
    gradient_descent

    Parameters

        x_init      initial guess
        objective   objective function
        gradient    gradient function
        eta         step size
        threshold   convergence threshold
        delta       numerical gradient estimate
        conv_by_grad true if converge by grad norm, false otherwise
        stochastic  true if stochastic updates, false by default
    """
    iterations = 0 # counter
    current_x = x_init # this one is updated
    n = x_init.shape[0]

    fx0 = 0 # last f(x)
    fx1 = float("inf") # current f(x)

    # grad_norms = []

    while True:
        i = iterations % n
        t = iterations # t, as in handout

        if stochastic:
            current_x, grad = update(gradient, current_x, eta, i, t)
        else:
            current_x, grad = update(gradient, current_x, eta)

        current_norm = np.linalg.norm(grad)
        # grad_norms.append(current_norm)

        # estimate gradient norm, gradient
        if stochastic:
            est_slope, est_grad = central_difference(objective, current_x, delta, i)
        else:
            est_slope, est_grad = central_difference(objective, current_x, delta)
        # calculate objective function

        if stochastic:
            fx1 = objective(current_x, i)
        else:
            fx1 = objective(current_x)

        if debug:
            print("Gradient norm: {}\nCurrent X: {}\nObjective function: {}\nEstimated next gradient: {}"\
            .format(current_norm, current_x, fx1, est_grad))
            print("Past objective function: {}\n".format(fx0))

        # check for convergence
        if conv_by_grad and converge_grad(grad, threshold):
            break

        elif not conv_by_grad and converge_delta_fx(fx0, fx1, threshold):
            break
        
        # update "past" objective function
        fx0 = fx1
        iterations += 1

    print("Converged after {} iterations\n".format(iterations))
    print("We updated to {}\n".format(current_x))

    # plt.plot(range(iterations),grad_norms,'o')

    return (current_x, fx1)

def update(gradient, x, eta, i = None, t = None):
    """
    update(gradient, n, current_x, eta, i) returns the
    new x value after updating.

    Parameters
        gradient    gradient function
        x           vector to be updated
        eta         constant step size
        i           optional, to specify stochastic
    """
    if i is not None:
        assert t is not None
        grad = gradient(x, i)
        eta = (eta + t) ** (-0.75) # adjust learning rate
    else:
        grad = gradient(x)

    x_new = x - eta * grad

    return (x_new, grad)

##################################
# Estimations
##################################

def central_difference(f, x, delta, stochI = None):
    """
    central_difference(f, x, delta) calculates an approximation of the gradient
    at a given point, for a given function f. The central difference is defined as:

        (f(x + delta/2) - f(x - delta/2)) / delta
    """
    n, m = x.shape

    delta_matrix = np.identity(n) * delta/2

    est_gradient = []
    for i in range(n):
        new_x_pos = x + delta_matrix[i].reshape(n, 1)
        new_x_neg = x - delta_matrix[i].reshape(n, 1)

        if stochI is not None:
            f_pos = f(new_x_pos, stochI)
            f_neg = f(new_x_neg, stochI)
        else:
            f_pos = f(new_x_pos)
            f_neg = f(new_x_neg)

        # print(f_pos - f_neg)

        est_gradient.append((f_pos - f_neg) / delta)

    return (np.linalg.norm(est_gradient), est_gradient)

def converge_grad(grad, threshold):
    """
    converge_grad(grad, threshold) returns True if the gradient's norm
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
