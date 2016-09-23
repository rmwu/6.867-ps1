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

def test_part1(is_gaussian, conv_grad_norm):
    """
    Parameters
        is_gaussian     True if use gaussian, False for quad bowl
        conv_grad_norm  True if we converge by gradient norm, False for
                        change in loss(x)
    """
    mu, Sigma, A, b = get_params()

    eta = 0.0001
    threshold = 0.02
    params = (mu, Sigma) if is_gaussian else (A, b)
    gradient = d_negative_gaussian if is_gaussian else d_quadratic_bowl

    x_init = np.random.rand(2, 1)
    x_init = np.array([[-100], [-100]])

    gradient_descent(x_init, params, least_square_error, gradient, 
                     eta, threshold, conv_grad_norm)

def main():
    test_part1(False, True)

if __name__ == "__main__":
    if debug:
        cProfile.run("main()")
    else:
        main()