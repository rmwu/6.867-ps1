"""
6.867 Fall 2016 - Problem Set 1
Problem 1. Implement Gradient Descent
"""
from __future__ import division

import numpy as np

from math_functions import *

debug = False

##################################
# Main Functions
##################################

def gradient_descent(x_init, objective, gradient, eta, threshold, delta, conv_by_grad):
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
    """
    iterations = 0 # counter
    current_x = x_init # this one is updated

    fx0 = 0 # last f(x)
    fx1 = float("inf") # current f(x)

    while True:
        current_x, grad = update(gradient, current_x, eta)
        current_norm = np.linalg.norm(grad)

        # estimate gradient norm, gradient
        est_slope, est_grad = central_difference(objective, current_x, delta)
        # calculate objective function
        fx1 = objective(current_x)

        if debug:
            print("Gradient norm: {}\nCurrent X: {}\nObjective function: {}\nEstimated gradient: {}"\
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

    return (current_x, fx1)

def update(gradient, x, eta):
    """
    update(gradient, n, current_x, eta) returns the
    new x value after updating.

    Parameters
        gradient    gradient function
        x           vector to be updated
        eta         constant step size
    """
    grad = gradient(x)
    x_new = x - eta * grad

    return (x_new, grad)

##################################
# Estimations
##################################

def central_difference(f, x, delta):
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

        f_pos = f(new_x_pos)
        f_neg = f(new_x_neg)

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
