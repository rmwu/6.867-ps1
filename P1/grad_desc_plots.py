"""
Test cases for illustrating effects of varying params on
gradient descent.
"""
import sys
import numpy as np 
import math
import matplotlib.pyplot as plt

from run_grad_descent import simple_gradient_descent, test_batch_gradient_descent

##################################
# Test Cases
##################################

def vary_eta(conv_type):
    etas = [1e-2, 1e-3, 1e-4] # eta
    threshold = 1e-3 # convergence threshold
    is_gaussian = False # quad bowl
    # converge by change in objective
    conv_by_grad = True if conv_type == "grad" else False

    outputs = [] # nested list

    for eta in etas:
        outputs.append(simple_gradient_descent(
            is_gaussian, conv_by_grad, eta, threshold))

    visualize(outputs, conv_type)

def vary_starting_guess(conv_type):
    eta = 1e-2
    threshold = 1e-1 # convergence threshold
    is_gaussian = False # quad bowl
    # converge by change in objective
    conv_by_grad = True if conv_type == "grad" else False
    guesses = [ np.array([[26], [26]]),
                np.array([[-100], [-100]]),
                np.array([[100], [100]])]

    outputs = [] # nested list

    for starting_guess in guesses:
        outputs.append(simple_gradient_descent(
            is_gaussian, conv_by_grad, eta, threshold, starting_guess = starting_guess))

    visualize(outputs, conv_type)

def vary_starting_guess(conv_type):
    eta = 1e-2
    threshold = 1e-1 # convergence threshold
    is_gaussian = False # quad bowl
    # converge by change in objective
    conv_by_grad = True if conv_type == "grad" else False
    guesses = [ np.array([[26], [26]]),
                np.array([[-100], [-100]]),
                np.array([[100], [100]])]

    outputs = [] # nested list

    for starting_guess in guesses:
        outputs.append(simple_gradient_descent(
            is_gaussian, conv_by_grad, eta, threshold, starting_guess = starting_guess))

    visualize(outputs, conv_type)

##################################
# Visualization with pyplot
##################################

def visualize(outputs, conv_type):
    colors = ["r", "b", "g", "o"]
    
    for output, color in list(zip(outputs, colors)):
        current_x, fx1, iterations, grad_norms, delta_obj = output

        delta_obj = list(map(math.log, [abs(delta) for delta in delta_obj]))
        y_values = grad_norms if conv_type == "grad" else delta_obj

        plt.plot(np.arange(iterations + 1), y_values, color=color)

    plt.xlabel("Number of iterations")

    y_axis_label = "Gradient norm"if conv_type == "grad" else \
    "Log absolute change in objective function"
    plt.ylabel(y_axis_label)

    plt.show()

##################################
# CLI
##################################

def main():
    if len(sys.argv) != 3:
        print("Must include choice of plot type and convergence criterion.")
        sys.exit(1)

    plot_type = sys.argv[1]
    conv_type = sys.argv[2]

    if plot_type == "eta":
        vary_eta(conv_type)
    elif plot_type == "guess":
        vary_starting_guess(conv_type)

if __name__ == "__main__":
    main()