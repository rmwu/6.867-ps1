import sys

import cProfile
import numpy as np
import math
# import matplotlib.pyplot as plt

from gradientDescent import *
from math_functions import *

debug = False

##################################
# Variations on gradient descent
##################################

def simple_gradient_descent(is_gaussian, conv_by_grad, eta, threshold, delta = 0.05,
        starting_guess = None):
    """
    Parameters
        is_gaussian     True if use gaussian, False for quad bowl
        conv_by_grad    True if we converge by gradient norm, False for
                        change in loss(x)
        eta             step size (0.0001 good for quad)
        threshold       convergence threshold
    """
    objective = negative_gaussian if is_gaussian else quadratic_bowl
    gradient = d_negative_gaussian if is_gaussian else d_quadratic_bowl

    # verify_gradient(objective, gradient, 1e-8)

    if starting_guess is None:
        starting_guess = np.array([9,9])

    return gradient_descent(starting_guess, objective, gradient, 
                     eta, threshold, delta, conv_by_grad)

def test_batch_gradient_descent(conv_by_grad, eta, threshold, delta = 0.05):
    """
    omg it converged for eta = 0.000001
    """
    theta_init = np.array([-5.2 for i in range(10)]).reshape(10,)

    # verify_gradient(squared_error, d_squared_error, 1e-8)

    return gradient_descent(theta_init, squared_error, d_squared_error,
                    eta, threshold, delta, conv_by_grad)

def test_stochastic_gradient_descent(conv_by_grad, eta, threshold, delta = 0.05):
    """
    eta 0.0001 threshold 0.05 sighhhh
    """
    theta_init = np.array([-5.2 for i in range(10)]).reshape(10,)

    # verify_gradient(stochastic_error, d_stochastic_error, 1e-8, True)

    return gradient_descent(theta_init, stochastic_error, d_stochastic_error,
                    eta, threshold, delta, conv_by_grad, True)

def test_verify_gradient():
    deltas = [1000, 100, 10, 1e-2, 1e-4, 1e-8]
    for delta in deltas:
        verify_gradient(squared_error, d_squared_error, delta)

##################################
# Numerical analysis
##################################

def verify_gradient(objective, gradient, delta, stoch = False):
    """
    Verifies the our gradient function.
    """
    dimensions = 10
    weight_vector = np.random.rand(dimensions)
    if stoch: 
        stochInd = np.random.randint(dimensions)
        approx_grad = central_difference(objective, weight_vector, delta, stochInd)
        exact_grad = gradient(weight_vector, stochInd)
    else:
        approx_grad = central_difference(objective, weight_vector, delta)
        exact_grad = gradient(weight_vector)

    grad_err = approx_grad - exact_grad
    mse = grad_err.T.dot(grad_err).mean()

    approx_grad_norm_sq = approx_grad.T.dot(approx_grad)

    print("verified gradient with central difference approximation, delta = {}".format(delta))
    print("delta is {} percent of the gradient norm sq".format(delta / math.sqrt(approx_grad_norm_sq)))
    print("mse is {} percent of the gradient norm sq".format(mse / approx_grad_norm_sq))
    print("exact gradient\n{}\napprox gradient\n{}\nmean sq err\n{}\n\n".format(exact_grad, approx_grad, mse))
    if mse / approx_grad_norm_sq > 1e-5:
        raise RuntimeError("bad gradient!")

##################################
# CLI implementation
##################################

def usage():
    raise ValueError("sys.argv must contain:\n'guass' or 'quad'\neta\nthreshold")

def main():
    if len(sys.argv) < 4:
        usage()

    # retrieve args
    func_name, eta, threshold = sys.argv[1:4]

    eta, threshold = float(eta), float(threshold)

    is_gaussian = False
    conv_by_grad = True

    plot = False
    if len(sys.argv) == 5 and sys.argv[4] == "plot":
        plot = True

    if len(sys.argv) == 6:
        delta = float(sys.argv[5])
    else:
        delta = 0.05

    # run appropriate function
    if func_name == 'batch':
        test_batch_gradient_descent(conv_by_grad, eta, threshold, delta)
    elif func_name == 'verify':
        test_verify_gradient()
    elif func_name == 'stoch':
        test_stochastic_gradient_descent(conv_by_grad, eta, threshold, delta)
    else:
        if func_name == 'gauss':
            is_gaussian = True
        elif func_name != 'quad':
            raise ValueError("must specify func_name")

        simple_gradient_descent(is_gaussian, conv_by_grad, eta, threshold, delta)

if __name__ == "__main__":
    if debug:
        cProfile.run("main()")
    else:
        main()