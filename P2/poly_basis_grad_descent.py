import sys
sys.path.append("..")

import numpy as np

from P1.gradientDescent import gradient_descent, central_difference
import lbf_regression as lbfr
import loadFittingDataP2 as lfd

def square_sum_error(y_guess, y_actual):
    """
    Sum of squares error, Bishop Equation 3.12.
    :return: 1/2 * sum of squared errors.
    """
    error_vec = y_guess - y_actual
    return 1/2 * error_vec.T.dot(error_vec)


def poly_basis_square_sum_error(weight_vector, X, Y):
    """
    Squared sum error, given a weight vector from a polynomial basis fit and
    data points (x, y) (stored in X and Y vectors)
    """
    assert (len(weight_vector.shape)==1 or weight_vector.shape[1]==1)
    degree = len(weight_vector) - 1
    Y_guess = lbfr.design_matrix(
        lbfr.power_basis(degree), X).dot(weight_vector)
    return square_sum_error(Y_guess, Y)


def poly_basis_square_sum_error_grad(weight_vector, X, Y):
    """
    Gradient of squared sum error, given a weight vector from a poly basis fit
    """
    degree = len(weight_vector) - 1
    phi = lbfr.design_matrix(lbfr.power_basis(degree), X)
    return phi.T.dot(phi.dot(weight_vector) - Y)


def poly_basis_square_sum_error_single_elt_grad(weight_vector, X, Y, i):
    """
    Same as above, but only the gradient on a single data point
    """
    degree = len(weight_vector) - 1
    single_data_point_x = np.array([X[i]])
    single_data_point_y = np.array([Y[i]])
    phi = lbfr.design_matrix(lbfr.power_basis(degree), single_data_point_x)
    return phi.T.dot(phi.dot(weight_vector) - single_data_point_y)


def verify_gradient():
    """
    Verifies the our gradient function.
    """

    num_datapoints = 5000
    degree = 4  # deg of polynomial fit

    # "observed" data points (fake data)
    data_amplitude = 1
    rand_x = data_amplitude * (np.random.rand(num_datapoints) - 0.5)
    rand_y = data_amplitude * (np.random.rand(num_datapoints, 1) - 0.5)
    weight_vector = 10 * (np.random.rand(degree, 1) - 0.5)

    def obj_func(weights):
        return poly_basis_square_sum_error(weights, rand_x, rand_y)
    approx_grad = central_difference(obj_func, weight_vector, 1e-8)
    exact_grad = poly_basis_square_sum_error_grad(weight_vector, rand_x, rand_y)

    grad_err = approx_grad - exact_grad
    mse = grad_err.T.dot(grad_err).mean()

    print("exact gradient\n{}\napprox gradient\n{}\nmean sq err\n{}\n\n".format(exact_grad, approx_grad, mse))
    if mse / approx_grad.T.dot(approx_grad) > 1e-5:
        raise RuntimeError("bad gradient!")


def main():
    # for i in range(int(sys.argv[1])):
    #     verify_gradient()

    poly_fit_degree = int(sys.argv[1])
    X_raw, Y_raw = lfd.getData(ifPlotData=False)

    assert len(X_raw) == len(Y_raw)
    num_datapoints = len(X_raw)
    X = np.array(X_raw).reshape(num_datapoints)
    Y = np.array(Y_raw).reshape(num_datapoints)


    bgd_weight_vector = gradient_descent(
        np.zeros(poly_fit_degree + 1),  # initial guess
        lambda w: poly_basis_square_sum_error(w, X, Y),  # obj function
        lambda w: poly_basis_square_sum_error_grad(w, X, Y),
        0.05, # step size
        1e-5, # convergence threshold
        0.05, # gradient approximation parameter (for central diff)
        False, # no convergence by grad norm
        stochastic=False
        )

    sgd_weight_vector = gradient_descent(
        np.zeros(poly_fit_degree + 1),  # initial guess
        lambda w, i: poly_basis_square_sum_error(w, np.array([X[i]]), np.array([Y[i]])),  # obj function
        lambda w, i: poly_basis_square_sum_error_grad(w, np.array([X[i]]), np.array([Y[i]])),
        0.01, # step size
        1e-5, # convergence threshold
        0.05, # gradient approximation parameter (for central diff)
        False, # no convergence by grad norm
        stochastic=True
        )



if __name__ == '__main__':
    main()