import sys

import cProfile
import numpy as np

from gradientDescent import *

debug = False

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