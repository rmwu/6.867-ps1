import sys

import cProfile
import numpy as np

import loadFittingDataP1 as loadData
import loadParametersP1 as loadParams

from gradientDescent import *

debug = False

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

##################################
# Part Testing
##################################

def test_part1(is_gaussian, conv_grad_norm, eta, threshold):
    """
    Parameters
        is_gaussian     True if use gaussian, False for quad bowl
        conv_grad_norm  True if we converge by gradient norm, False for
                        change in loss(x)
        eta             step size
        threshold       convergence threshold
    """
    mu, Sigma, A, b = get_params()
    mu = np.array(mu).reshape(2, 1)

    b = np.array(b).reshape(2, 1)

    # eta = 1000  # step size
    # threshold = 1e-10  # convergence threshold for gradient norm
    params = (mu, Sigma) if is_gaussian else (A, b)
    function = negative_gaussian if is_gaussian else quadratic_bowl
    gradient = d_negative_gaussian if is_gaussian else d_quadratic_bowl

    x_init = np.random.rand(2, 1)
    x_init = np.array([[9.999], [9.99]])

    gradient_descent(x_init, params, least_square_error, function, gradient, 
                     eta, threshold, conv_grad_norm)

def test_part2(is_gaussian, conv_grad_norm):
    """
    Parameters
        is_gaussian     True if use gaussian, False for quad bowl
        conv_grad_norm  True if we converge by gradient norm, False for
                        change in loss(x)
    """
    delta = 0.05

    mu, Sigma, A, b = get_params()

    eta = 3
    threshold = 1e-10
    params = (mu, Sigma) if is_gaussian else (A, b)
    function = negative_gaussian if is_gaussian else quadratic_bowl
    gradient = d_negative_gaussian if is_gaussian else d_quadratic_bowl

    x_init = np.random.rand(2, 1)
    x_init = np.array([100, 100])

    gradient_descent(x_init, params, least_square_error, function, gradient, 
                     eta, threshold, conv_grad_norm, delta)

def usage():
    raise ValueError("sys.argv must contain:\n'guass' or 'quad'\neta\nthreshold")

def main():
    if len(sys.argv) != 4:
        usage()

    func_name, eta, threshold = sys.argv[1:4]
    eta, threshold = float(eta), float(threshold)

    print("eta: {}\nthreshold: {}".format(eta, threshold))

    is_gaussian = False
    conv_grad_norm = True

    if func_name == 'gauss':
        is_gaussian = True
    elif func_name != 'quad':
        raise ValueError("must specify 'gauss' or 'quad'")

    test_part1(is_gaussian, conv_grad_norm, eta, threshold)

if __name__ == "__main__":
    if debug:
        cProfile.run("main()")
    else:
        main()