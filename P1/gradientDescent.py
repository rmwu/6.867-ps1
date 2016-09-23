"""
6.867 Fall 2016 - Problem Set 1
Problem 1. Implement Gradient Descent
"""

import numpy as np
import math
import cProfile

import loadFittingDataP1 as loadData
import loadParametersP1 as loadParams

debug = True

##################################
# Main Functions
##################################

def gradient_descent(objective_function, gradient_function, 
                     eta, threshold, grad_norm = False):
    """
    Parameters
        objective_function
            Function used to calculate how good our guess
            is so far. In this case, our objective_function
            is a loss function to be minimized.
        gradient_function
            Function used to calculate the gradient at a given
            point.
        eta
            constant step size
        threshold
            convergence threshold
        grad_norm
            True if we compute change in gradient norm, False if
            we compute change in the vector
    """
    if grad_norm:
        pass

def update(gradient, x, eta):
    """
    update(gradient, x, eta) returns the new x
    value after xxxxxx

    Parameters
        gradient   gradient function
        x           vector to be updated
        eta         constant step size

    TODO kwargs
    """
    return x - eta * gradient(x)

##################################
# Mathematical Implementations
##################################

def d_negative_gaussian(Sigma, x, mu, n):
    """
    d_negative_gaussian(Sigma, x, mu, n) calculates the derivative
    of a negative gaussian function, which has the form:

        f(x) = [math]

    Parameters
        Sigma   covariance matrix
        x       x is x
        n       n is data size
        mu      mean
    """
    f_x = negative_gaussian(Sigma, x, mu, n)

    return -f_x * np.linalg.inv(Sigma).dot(x - mu)

def negative_gaussian(Sigma, x, mu, n):
    """
    negative_gaussian(Sigma, x, mu, n) calculates the negative gaussian
    function given inputs.
    """
    scaling_factor = 1/(math.sqrt(math.pi) ** n * np.linalg.det(Sigma))
    exponent = -.5 * reduce(np.dot, [(x - mu).T, np.linalg.inv(Sigma), (x - mu)])

    return math.exp(exponent)

def d_quadratic_bowl(A, x, b):
    """
    d_quadratic_bowl(A, x, b) calculates the derivative of a 
    quadratic bowl, which has the form:

        f(x) = 1/2 xT Ax - xT b

    Parameters
        A       positive definite matrix
        x
        b
    """
    return A.dot(x) - b

def quadratic_bowl(A, x, b):
    """
    quadratic_bowl(A, x, b) calculates the quadratic bowl function for given
    A, x, and b.
    """
    return 0.5 * x.T.dot(A).dot(x) - x.T.dot(b)

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
    """
    return numpy.linalg.norm(grad) < threshold

def converge_delta_fx(fx0, fx1, threshold):
    """
    converge_delta_fx(fx0, fx1, threshold) returns True if the change in f(x)
    between two success updates is less than threshold, False otherwise.
    """
    return abs(fx1 - fx0) < threshold

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
    X, y = get_inputs()
    mu, Sigma, A, b = get_params()


if __name__ == "__main__":
    if debug:
        cProfile.run("main()")
    else:
        main()
