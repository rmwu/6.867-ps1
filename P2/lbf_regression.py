import sys
sys.path.append("..")

import math
import numpy as np
import matplotlib.pyplot as plt

import P2.loadFittingDataP2 as lfd


def power_basis(degree):
    def power_func(k):
        return lambda x: x**k
    return [power_func(k) for k in range(degree+1)]


def cosine_basis(degree):
    def cosine_func(k):
        return lambda x: math.cos(k*math.pi*x)
    return [cosine_func(k) for k in range(degree+1)]


def basis_fit(X, Y, basis_functions):
    """
    Perform a maximum-likelihood linear basis-function-fit with specified
    basis_functions.
    """
    if len(X) != len(Y):
        raise ValueError("X and Y must have same dimensions n x 1")

    num_datapoints = len(X)
    y_values = Y.reshape(num_datapoints, 1)

    design_matrix = np.array([[f(x) for f in basis_functions] for x in X])
    pseudoinverse = np.linalg.inv(design_matrix.T.dot(design_matrix)).dot(design_matrix.T)
    max_likelihood_weights = pseudoinverse.dot(y_values)

    assert len(max_likelihood_weights) == len(basis_functions)
    return max_likelihood_weights


def basis_fit_plot(X, Y, basis_functions, x_arange, **kwargs):
    """
    Plot the maximum likelihood linear basis function regression.
    """
    x_arange_design_matrix = np.array([[f(x) for f in basis_functions] for x in x_arange])
    max_likelihood_weights = basis_fit(X, Y, basis_functions)
    print("maximum likelihood weights for basis\n{}\n".format(max_likelihood_weights))
    y_fit_values = x_arange_design_matrix.dot(max_likelihood_weights)
    plt.plot(x_arange, y_fit_values, **kwargs)


def design_matrix(basis_functions, X):
    """
    The design matrix
    [[f0(x0), f1(x0), ...],
     [f0(x1), f1(x1), ...],
     ...]
    """
    return np.array([[f(x) for f in basis_functions] for x in X])


def visualize(fit_degree, poly_basis):
    X, Y = lfd.getData(ifPlotData=False)

    # plot the points
    plt.plot(X,Y,'o')

    x_arange = np.arange(0.0, 1.0, 0.005)

    # plot the actual function
    y_actual_values = np.cos(math.pi * x_arange) + np.cos(2 * math.pi * x_arange)
    line, = plt.plot(x_arange, y_actual_values, lw=2, color="y")

    # plot the fit
    fit_kwargs = {"color": 'r', "lw": 2}
    basis_functions = power_basis(fit_degree) if poly_basis else cosine_basis(fit_degree)
    basis_fit_plot(X, Y, basis_functions, x_arange, **fit_kwargs)

    # nice setup
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()


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
    visualize(fit_degree, poly_basis)


if __name__ == '__main__':
    main()
    