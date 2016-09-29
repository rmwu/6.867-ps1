"""
math_functions contains the mathematical implementations of
objective functions and their gradients
"""
import numpy as np
import math

import loadFittingDataP1 as loadData
import loadParametersP1 as loadParams

##################################
# Input handling
##################################

def get_gaussian_params():
    mu, sigma = loadParams.getData()[:2]
    mu = np.array(mu).reshape(2)
    sigma = np.array(sigma)

    return (mu, sigma)

def get_quad_params():
    a, b = loadParams.getData()[2:4]
    b = np.array(b).reshape(2)
    a = np.array(a)

    return (a, b)

def get_data():
    x, y = loadData.getData()
    x, y = np.array(x).reshape(100, 10), np.array(y).reshape(100)
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

