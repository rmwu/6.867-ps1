"""
6.867 Fall 2016 - Problem Set 1
Problem 1. Implement Gradient Descent
"""

import numpy as np
import math

##################################
# Main Functions
##################################

def gradient_descent(x_init, params, loss, gradient, 
                     eta, threshold, grad_norm = False):
    """
    Parameters
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
    p1, p2 = params

    current_x = x_init
    current_loss = 0

    while True:
        current_x, grad = update(gradient, params, n, current_x, eta)
        # update current_loss

        current_norm = np.linalg.norm(grad)

        print("Gradient norm: {} | Current X: {}".format(current_norm, current_x))

        if grad_norm and converge_grad_norm(grad, threshold):
            break

        elif not grad_norm and converge_delta_fx(prev_loss, current_loss, threshold):
            break

        elif current_norm > 1000:
            break

        # update previous loss

    return (current_x, current_loss)


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

    return (grad, x_new)

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
        params      contains (mu, Sigma)
        n           number of samples
        x           current vector
    """
    mu, Sigma = params

    f_x = negative_gaussian(params, n, mu)

    n, m = (x - mu).shape
    # return -f_x * np.linalg.inv(Sigma).dot((x - mu).reshape(m, n))
    return -f_x * np.linalg.inv(Sigma).dot(x - mu)

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

    scaling_factor = 1/(math.sqrt(math.pi) ** n * np.linalg.det(Sigma))
    exponent = -.5 * (x - mu).T.dot(np.linalg.inv(Sigma)).dot(x - mu)

    return math.exp(exponent)

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

# error

def least_square_error(X, Theta, y):
    """
    least_square_error(X, Theta, y) returns the least square error, defined
    as:

        J(Theta) = || X Theta - y || ^2
    """
    pass

def central_difference(f, x, delta):
    """
    central_difference(f, x, delta) calculates an approximation of the gradient
    at a given point, for a given function f. The central difference is defined as:

        (f(x + delta/2) - f(x - delta/2)) / delta

    TODO dimensions

    TODO kwargs
    """
    return (f(x + delta/2) - f(x - delta/2)) / delta

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
