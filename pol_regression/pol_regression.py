import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# ---------- POLYNOMIAL REGRESSION FUNCTIONS ----------

# Create the matrix for values of x based on the degree (if degree = 2 then matrix will have size of 2)
def get_data_matrix(x, degree):
    matrix = np.ones(x.shape)

    # Increment size of the matix depending on degree
    for i in range(1, degree + 1):
        matrix = np.column_stack((matrix, x ** i))

    return matrix

def get_weight_matrix(x, y, degree):
    # Calculate the weight for each polynomial
    X = get_data_matrix(x, degree)
    XX = X.transpose().dot(X)

    return np.linalg.solve(XX, X.transpose().dot(y))

# Main regression function
def pol_regression(features_train, y_train, degree):
    w = get_weight_matrix(features_train, y_train, degree)
    return w


# ---------- EVALUATION ----------

# Make prediction using test data
def prediction(x_test, w, degree):
    x = get_data_matrix(x_test, degree)
    y = x.dot(w)
    return y

# Calculate the RMSE of the polynomial regression model
def eval_pol_regression(y_hat, w, x, y, degree):
    # Predict test values for y
    error = 0
    for i, pred in enumerate(y_hat):
        # Calculate difference between predicted value and actual value
        residual = y[i] - pred

        # Get squared difference
        error += residual ** 2

    rmse = math.sqrt(error / len(y_hat))
    return rmse


# ---------- VISUALISTATION ----------

def plot_error_graph(train_error, test_error):
    plt.figure()
    plt.semilogy(range(1,8), train_error)
    plt.semilogy(range(1,8), test_error)
    plt.legend(('RMSE on training set', 'RMSE on test set'))
    plt.savefig(os.path.join('images','polynomial_evaluation.png'))
