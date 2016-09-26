import sys

import cProfile
import numpy as np

import loadFittingDataP1 as loadData
import loadParametersP1 as loadParams

from gradientDescent import *

debug = False

##################################
# Part Testing
##################################

def simple_gradient_descent(is_gaussian, conv_by_grad, eta, threshold, delta = 0.05):
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

    x_init = np.array([[26], [26]])

    gradient_descent(x_init, function, gradient, 
                     eta, threshold, delta, conv_by_grad)

def test_batch_gradient_descent(conv_by_grad, eta, threshold, delta = 0.05):
    """
    omg it converged for eta = 0.000001
    """
    theta_init = np.array([0 for i in range(10)]).reshape(10, 1)

    gradient_descent(theta_init, squared_error, d_squared_error,
                    eta, threshold, delta, conv_by_grad)

def test_stochastic_gradient_descent(conv_by_grad, eta, threshold, delta = 0.05):
    """
    omg it converged for eta = 0.000001
    """
    theta_init = np.array([-5.3 for i in range(10)]).reshape(10, 1)

    gradient_descent(theta_init, stochastic_error, d_stochastic_error,
                    eta, threshold, delta, conv_by_grad, True)

def usage():
    raise ValueError("sys.argv must contain:\n'guass' or 'quad'\neta\nthreshold")

def main():
    if len(sys.argv) < 4:
        usage()

    # retrieve args
    func_name, eta, threshold = sys.argv[1:4]

    if len(sys.argv) == 5:
        delta = float(sys.argv[4])
    else:
        delta = 0.05

    eta, threshold = float(eta), float(threshold)

    is_gaussian = False
    conv_by_grad = False

    if func_name == 'gauss':
        is_gaussian = True
    elif func_name != 'quad':
        raise ValueError("must specify 'gauss' or 'quad'")

    # simple_gradient_descent(is_gaussian, conv_by_grad, eta, threshold, delta)
    # test_stochastic_gradient_descent(conv_by_grad, eta, threshold, delta)
    test_batch_gradient_descent(conv_by_grad, eta, threshold, delta = 0.05)

if __name__ == "__main__":
    if debug:
        cProfile.run("main()")
    else:
        main()