import sys
sys.path.append("..")

import numpy as np

import P1.gradientDescent as gd
import lbf_regression as lbfr

def square_sum_error(y_guess, y_actual):
    """
    Sum of squares error, Bishop Equation 3.12.
    :return: 1/2 * sum of squared errors.
    """
    error_vec = y_guess - y_actual
    return 1/2 * error_vec.T.dot(error_vec)


def poly_basis_square_sum_error(weight_vector, X, Y):
    """
    Squared sum error, given a weight vector from a polynomial basis fit
    """
    degree = len(weight_vector) - 1
    return square_sum_error(lbfr.design_matrix(power_basis(degree), X), Y)


def poly_basis_square_sum_error_grad(weight_vector, X, Y):
    """
    Gradient of squared sum error, given a weight vector from a poly basis fit
    """
    assert len(Y.shape) == 2
    assert Y.shape[1] == 1
    degree = len(weight_vector) - 1
    phi = lbfr.design_matrix(power_basis(degree), X)
    return phi.T.dot(phi.dot(weight_vector) - Y)


def verify_gradient():
    num_datapoints = 100
    degree = 8

    rand_x = 20 * (np.random.rand(num_datapoints) - 0.5)
    rand_y = 20 * (np.random.rand(num_datapoints) - 0.5)
    weight_vector = 10 * (np.random.rand(degree, 1) - 0.5)

    def obj_func(weights):
        return poly_basis_square_sum_error(weights, rand_x, rand_y)
    approx_grad = gd.central_difference(obj_func, weight_vector, 1e-8)
    exact_grad = poly_basis_square_sum_error_grad(weight_vector, rand_x, rand_y)
    grad_err = approx_grad - exact_grad
    mse = grad_err.T.dot.grad_err.mean()

    print("exact gradient\n{}\napprox gradient\n{}\nmean sq err\n{}".format(approx_grad, exact_grad, mse))
    if mse / approx_grad.T.dot.approx_grad > 1e-5:
        raise RuntimeError("bad gradient!")


def main():
    """
    Verifies the our gradient function.
    """
    for i in range(int(sys.argv[1])):
        verify_gradient()


if __name__ == '__main__':
    main()