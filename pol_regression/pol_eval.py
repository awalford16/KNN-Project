import data_processing as dp
import matplotlib.pyplot as plt
import pol_regression as pol
import os

def main():
    # Data processing
    df = dp.load_data('pol_regression.csv')

    x_train, y_train, x_test, y_test = dp.split_data(df)

    # Create an array to represent the different test errors of each degree
    train_error = []
    test_error = []

    # Plot ground truth
    plt.figure()
    plt.ylim(0,1.5)
    #plt.plot(x_train, y_train, 'bo')
    plt.plot(x_test, y_test, 'bo')

    colors = ['r', 'y', 'b', 'c', 'k', 'm', 'g']

    # Perform polynomial regression for powers 0 to 10
    for i, degree in enumerate([0, 1, 2, 3, 4, 5, 10]):
        w = 1

        if degree != 0:
            # Calculate the coefficients based on the training values
            w = pol.pol_regression(x_train, y_train, degree)

        # Make predictions for test data
        y_train_hat = pol.prediction(x_train, w, degree)
        y_test_hat = pol.prediction(x_test, w, degree)

        # Plot predictions
        list = zip(*sorted(zip(*(x_test, y_test_hat))))
        plt.plot(*list, color=colors[i])

        # Measure accuracy of model
        # RMSE of training set
        train_error.append(pol.eval_pol_regression(y_train_hat, w, x_train, y_train, degree))

        # RMSE of testing set
        test_error.append(pol.eval_pol_regression(y_test_hat, w, x_test, y_test, degree))


        print("[Degree: {0}] - Train: {1:.4f}, Test: {2:.4f}".format(degree, train_error[i], test_error[i]))


    plt.legend(('ground truth', '$x^0$', '$x$', '$x^2$', '$x^3$', '$x^4$', '$x^5$', '$x^{10}$'), loc = 'lower right')
    plt.savefig(os.path.join('images', 'polynomial_split.png'))

    pol.plot_error_graph(train_error, test_error)

if __name__ == '__main__':
    main()