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
def eval_pol_regression(w, x, y, degree):
    # Predict test values for y
    y_hat = prediction(x, w, degree)
    error = 0
    for i, pred in enumerate(y_hat):
        # Calculate difference between predicted value and actual value
        residual = y[i] - pred

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

def plot_error_graph(train_error, test_error):
    plt.figure()
    plt.semilogy(range(1,11), train_error)
    plt.semilogy(range(1,11), test_error)
    plt.legend(('RMSE on training set', 'RMSE on test set'))
    plt.savefig('polynomial_evaluation.png')
    plt.show()


# ---------- PROGRAM ----------

def main():
    # Data processing
    df = load_data()
    x_train, y_train, x_test, y_test = split_data(df)

    # Perform polynomial regression for powers 0 to 10
    for i in [0, 1,2,3,4,5,10]:
        if i != 0:
            # Calculate the coefficients based on the training values
            w = pol_regression(x_train, y_train, i)
        else: 
            w = 1

        # Measure accuracy of model
        # RMSE of training set
        train_error = eval_pol_regression(w, x_train, y_train, i)

        # RMSE of testing set
        test_error = eval_pol_regression(w, x_test, y_test, i)
        print("[Degree: {0}] - Train: {1:.4f}, Test: {2:.4f}".format(i, train_error, test_error))



if __name__ == "__main__":
    main()