import sys

import math
import numpy as np
import matplotlib.pyplot as plt

import loadFittingDataP2 as lfd

def powers(x, degree):
    """
    :param: x: a real number
    :param: degree: a nonnegative integer
    :return: the list [1, x, x^2, ..., x^degree]
    """
    # check preconditions
    if type(degree) is not int:
        raise TypeError("degree must be integer")
    if degree < 0:
        raise ValueError("degree must be nonnegative integer")

    return [x**power for power in range(degree + 1)]
    

def polynomial_basis_fit(X, Y, fit_degree):
    """
    Performs linear basis function regression on pairs (x, y) (where both are
    real numbers) with a polynomial basis of max `fit_degree`. See Bishop,
    Equation 3.15

    :param: X: the values of x
    :param: Y: the values of y, in the same order as their counterparts in `X`
    :param: fit_degree: the maximum degree of the polynomial fit, so our basis
                        functions are, as functions of x, 1, x, x^2, ..., up
                        to x^`fit_degree`
    :return: the maximum likelihood weight vector
    """
    # preconditions
    if len(X) != len(Y):
        raise ValueError("X and Y must have same dimensions n x 1")

    num_datapoints = len(X)
    y_values = Y.reshape(num_datapoints, 1)

    x_with_powers = [powers(x, fit_degree) for x in X]  # design matrix
    phi = np.array(x_with_powers)
    pseudoinverse = np.linalg.inv(phi.T.dot(phi)).dot(phi.T)
    max_likelihood_weights = pseudoinverse.dot(y_values)

    assert len(max_likelihood_weights) == fit_degree + 1
    return max_likelihood_weights


def poly_fit_plot(X, Y, fit_degree, x_arange, **kwargs):
    """
    Plot the maximum likelihood polynomial basis function fit of degree
    fit_degree in red, linewidth 2
    """
    x_arange_powers = np.array([powers(x, fit_degree) for x in x_arange])
    max_likelihood_weights = polynomial_basis_fit(X, Y, fit_degree)
    y_fit_values = x_arange_powers.dot(max_likelihood_weights)
    line, = plt.plot(x_arange, y_fit_values, **kwargs)


def cosines(x, degree):
    """
    :return: cosines [cos(pi x), cos(2pi x), ..., cos(degree*pi x)]
    """
    return np.cos(np.arange(1, degree+1) * math.pi * x)


def cosine_basis_fit(X, Y, fit_degree):
    """
    Performs linear basis function regression on pairs (x, y) (where both are
    real numbers) with a cosine basis. (cos(pi*x), cos(2*pi*x), ... cos(M*pi*x))

    :param: X: the values of x
    :param: Y: the values of y, in the same order as their counterparts in `X`
    :param: fit_degree: the maximum degree of the cosine fit, so our basis
                        functions go up to cos(fit_degree*pi*x)
    :return: the maximum likelihood weight vector
    """
    # preconditions
    if len(X) != len(Y):
        raise ValueError("X and Y must have same dimensions n x 1")


    num_datapoints = len(X)
    y_values = Y.reshape(num_datapoints, 1)

    x_cosines = [cosines(x, fit_degree) for x in X]
    phi = np.array(x_cosines)
    pseudoinverse = np.linalg.inv(phi.T.dot(phi)).dot(phi.T)
    max_likelihood_weights = pseudoinverse.dot(y_values)

    assert len(max_likelihood_weights) == fit_degree
    return max_likelihood_weights


def cos_fit_plot(X, Y, fit_degree, x_arange, **kwargs):
    """
    Plot the maximum likelihood cosine basis function fit with
    basis functions cos(pi x), cos(2 pi x), ... , cosin(M pi x)
    where M = fit_degree
    """
    x_arange_cosines = np.array([cosines(x, fit_degree) for x in x_arange])
    max_likelihood_weights = cosine_basis_fit(X, Y, fit_degree)
    print("maximum likelihood weights for cosine basis\n{}\n".format(max_likelihood_weights))
    y_fit_values = x_arange_cosines.dot(max_likelihood_weights)
    plt.plot(x_arange, y_fit_values, **kwargs)


def main():
    if len(sys.argv) != 3:
        raise ValueError("must give basis name ('poly' or 'cos') and fit degree")

    basis_name = sys.argv[1]
    fit_degree = int(sys.argv[2])
    poly_basis = True
    if basis_name == 'cos':
        poly_basis = False
    elif basis_name != 'poly':
        raise ValueError("basis name must be 'poly' or 'cos'")

    X, Y = lfd.getData(ifPlotData=False)

    # plot the points
    plt.plot(X,Y,'o')

    x_arange = np.arange(0.0, 1.0, 0.005)

    # plot the actual function
    y_actual_values = np.cos(math.pi * x_arange) + np.cos(2 * math.pi * x_arange)
    line, = plt.plot(x_arange, y_actual_values, lw=2, color="y")

    # plot the fit
    fit_kwargs = {"color": 'r', "lw": 2}
    plot_func = poly_fit_plot if poly_basis else cos_fit_plot
    plot_func(X, Y, fit_degree, x_arange, **fit_kwargs)

    # nice setup
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    main()
    