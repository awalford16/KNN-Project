import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import random

# ---------- POLYNOMIAL REGRESSION FUNCTIONS ----------

# Create the matrix for values of x based on the degree (if degree = 2 then matrix will have size of 2)
def get_data_matrix(x, degree):
    matrix = np.ones(x.shape)

    # Increment size of the matix depending on degree
    for i in range(1, degree + 1):
        matrix = np.column_stack((matrix, x ** i))

    return matrix

def get_weight_matrix(x, y, degree):
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
def eval_pol_regression(x, y, y_hat, degree):
    error = 0
    for i, pred in enumerate(y_hat):
        # Calculate difference between predicted value and actual value
        residual = pred - y[i]

        # Get squared difference
        error += residual ** 2

    # Calculate mean and take square root of error
    rmse = math.sqrt(error / len(y_hat))
    return rmse


#  -------- DATA PROCESSING ------------

# Load the data from CSV file
def load_data():
    print("Loading in CSV file...")
    return pd.read_csv(os.path.join('data', 'pol_regression.csv'))

# Split data into train and test
def split_data(data):
    print('Splitting data...')
    # Normalise the data
    data=((data-data.min())/(data.max()-data.min()))

    # Shuffle data
    data = data.sample(frac=1)

    # Split data at 70%
    percent = int((data.shape[0]) * 0.7)
    train = data[:percent]
    test = data[percent:]

    # Return splits of data set
    return train['x'].values, train['y'].values, test['x'].values, test['y'].values


# ---------- VISUALISTATION ----------

def select_color(i):
    colors = ['r', 'g', 'c', 'm', 'y', 'b', 'k']

    return colors[i]

# Plot error graph
def plot_error_graph(train_error, test_error):
    plt.figure()
    plt.semilogy([0,1,2,3,4,5,10], train_error)
    plt.semilogy([0,1,2,3,4,5,10], test_error)
    plt.legend(('RMSE on training set', 'RMSE on test set'))
    plt.savefig('polynomial_error.png')
    plt.show()


# ---------- PROGRAM ----------

def main():
    # Data processing
    df = load_data()
    x_train, y_train, x_test, y_test = split_data(df)

    train_error_array = np.zeros((7, 1)) 
    test_error_array = np.zeros((7, 1)) 

    # Build graph for plotting
    plt.figure()

    # Plot training and testing data
    plt.plot(x_test, y_test, 'g')
    plt.plot(x_train, y_train, 'bo')

    # Perform polynomial regression for powers 0 to 10
    for i, degree in enumerate([0, 1,2,3,4,5,10]):
        if degree != 0:
            # Calculate the coefficients based on the training values
            w = pol_regression(x_train, y_train, degree)
        else: 
            w = 1

        # Make predictions based on training model
        y_hat_train = prediction(x_train, w, degree)
        y_hat_test = prediction(x_test, w, degree)

        # Measure accuracy of model
        # RMSE of training set
        train_error_array[i] = eval_pol_regression(x_train, y_train, y_hat_train, i)

        # RMSE of testing set
        test_error_array[i] = eval_pol_regression(x_test, y_test, y_hat_test, i)
        # print("[Degree: {0}] - Train: {1:.4f}, Test: {2:.4f}".format(i, train_error, test_error))

        # Plot predictions
        plt.plot(x_test, y_hat_test, select_color(i))
    
    
    # Set plot between 5 and -5
    plt.ylim((-5, 5))
    plt.xlim(-5,5)

    plt.legend(('Training Points', 'Actual Values', '$x^{0}$', '$x^{1}$', '$x^{2}$', '$x^{3}$', '$x^{4}$', '$x^{5}$', '$x^{10}$'), loc = 'lower left')
    plt.savefig('pol_regression.png')

    # Plot error graph
    plot_error_graph(train_error_array, test_error_array)


if __name__ == "__main__":
    main()