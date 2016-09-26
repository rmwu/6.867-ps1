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
    function = negative_gaussian if is_gaussian else quadratic_bowl
    gradient = d_negative_gaussian if is_gaussian else d_quadratic_bowl

    if starting_guess is None:
        starting_guess = np.array([[0], [0]])

    return gradient_descent(starting_guess, function, gradient, 
                     eta, threshold, delta, conv_by_grad)

def test_batch_gradient_descent(conv_by_grad, eta, threshold, delta = 0.05):
    """
    omg it converged for eta = 0.000001
    """
    theta_init = np.array([0 for i in range(10)]).reshape(10, 1)

    return gradient_descent(theta_init, squared_error, d_squared_error,
                    eta, threshold, delta, conv_by_grad)

def test_stochastic_gradient_descent(conv_by_grad, eta, threshold, delta = 0.05):
    """
    doesn't work
    """
    theta_init = np.array([-5.3 for i in range(10)]).reshape(10, 1)

    return gradient_descent(theta_init, stochastic_error, d_stochastic_error,
                    eta, threshold, delta, conv_by_grad, True)

##################################
# Visualization with pyplot
##################################

# def visualize(iterations, grad_norms):
#     # plot gradient over time
#     plt.plot(np.arange(iterations + 1),grad_norms, "")
#     plt.xlabel("iterations")
#     plt.ylabel("gradient norm")

#     plt.show()

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
    conv_by_grad = False

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
    else:
        if func_name == 'gauss':
            is_gaussian = True
        elif func_name != 'quad':
            raise ValueError("must specify 'gauss', 'quad', or 'batch'")

        simple_gradient_descent(is_gaussian, conv_by_grad, eta, threshold, delta)

    # if plot:
    #     visualize(iterations, grad_norms)
    # test_stochastic_gradient_descent(conv_by_grad, eta, threshold, delta)

if __name__ == "__main__":
    if debug:
        cProfile.run("main()")
    else:
        main()